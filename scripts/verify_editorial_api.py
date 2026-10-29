"""Real-model HTTP integration checks. No dataset/model/search responses mocked."""
from __future__ import annotations
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import sys
import threading
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import requests
from PIL import Image
import serve_editorial as api
from src.config import SETTINGS
from src.nlp.review_analyzer import top_review_evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expect', choices=['READY', 'FALLBACK'], required=True)
    args = parser.parse_args()
    report = {'expected_status': args.expect, 'offline': SETTINGS.offline, 'checks': {}, 'responses': {}}
    server = api.HopGuServer(('127.0.0.1', 0), api.HopGuRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    output = ROOT/'integration_verification'
    output.mkdir(exist_ok=True)
    calls = Counter()
    try:
        response = requests.get(base+'/api/status', timeout=300)
        response.raise_for_status()
        status = response.json()
        report['status'] = status
        assert status['backend_status'] == args.expect, status
        assert status['fusion_alpha'] == 0.8
        report['checks']['status_and_alpha'] = True
        service = api._get_service()
        # Spies delegate to the actual model/service; they never supply results.
        def spy(owner, name):
            original = getattr(owner, name)
            def wrapped(*a, **kw):
                calls[name] += 1
                return original(*a, **kw)
            setattr(owner, name, wrapped)
        for name in ('search_text', 'search_image', 'search_multimodal'):
            spy(service, name)
        spy(service.text_encoder, 'encode_query')
        spy(service.image_encoder, 'encode_image')
        spy(service.llm, '_chat_json')
        catalog_response = requests.get(base+'/api/catalog', timeout=30)
        catalog_response.raise_for_status()
        catalog = catalog_response.json()['products']
        assert len(catalog) == len(service.products) == 860
        by_id = {str(item['product_id']): item for item in catalog}
        target = by_id['271312499']
        image_response = requests.get(base+target['image_url'], timeout=10)
        assert image_response.status_code == 200 and image_response.headers['Content-Type'] == 'image/jpeg'
        image_bytes = image_response.content
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
        report['checks']['catalog_860_local_image'] = True
        category = target['category']
        filtered = requests.get(base+'/api/catalog', params={'category': category, 'max_price': 1000000, 'sort':'price_asc'}, timeout=30).json()['products']
        assert filtered and all(p['category'] == category and p['price_vnd'] <= 1000000 for p in filtered)
        assert [p['price_vnd'] for p in filtered] == sorted(p['price_vnd'] for p in filtered)
        report['checks']['catalog_filters_sort'] = True

        def post(fields, image=None, filename='271312499.jpg'):
            files = {key: (None, str(value)) for key, value in fields.items()}
            if image is not None: files['image'] = (filename, image, 'image/jpeg')
            return requests.post(base+'/api/recommend', files=files, timeout=600)

        def verify_unsupported_media_type():
            """Verify the 415 contract without treating one Windows TCP reset as an API crash."""
            attempts = []
            for attempt in range(1, 4):
                try:
                    # Send the protocol being rejected explicitly. A fresh session and
                    # Connection: close avoid reusing a socket from the preceding upload
                    # boundary tests on Windows.
                    with requests.Session() as protocol_session:
                        response = protocol_session.post(
                            base + '/api/recommend',
                            data=b'{"query":"x"}',
                            headers={
                                'Content-Type': 'application/json',
                                'Connection': 'close',
                            },
                            timeout=10,
                        )
                    payload = response.json()
                    attempts.append({'attempt': attempt, 'status_code': response.status_code})
                    assert response.status_code == 415, response.text
                    assert payload['status'] == 'error', payload
                    assert payload['error']['code'] == 'UNSUPPORTED_MEDIA_TYPE', payload
                    health = requests.get(base + '/api/status', timeout=30)
                    health.raise_for_status()
                    attempts[-1]['server_alive_after_response'] = True
                    report['responses']['unsupported_media_type'] = {
                        'attempts': attempts,
                        'payload': payload,
                    }
                    report['checks']['unsupported_content_type_415_server_alive'] = True
                    return
                except requests.ConnectionError as error:
                    health = requests.get(base + '/api/status', timeout=30)
                    health.raise_for_status()
                    attempts.append({
                        'attempt': attempt,
                        'transport_error': type(error).__name__,
                        'server_alive_after_error': True,
                    })
                    if attempt == 3:
                        report['responses']['unsupported_media_type'] = {'attempts': attempts}
                        raise

        for mode, query, upload in [('TEXT','tai nghe bluetooth',None), ('IMAGE','',image_bytes), ('MULTIMODAL','tai nghe bluetooth',image_bytes)]:
            before = calls.copy()
            start = perf_counter()
            response = post({'query':query,'top_k':3}, upload)
            response.raise_for_status()
            data = response.json()
            elapsed = (perf_counter()-start)*1000
            assert data['mode'] == mode and len(data['results']) == 3
            expected_request_state = 'FALLBACK' if data['metadata'].get('llm_reason_error') else args.expect
            assert data['backend_status'] == expected_request_state, data
            if data['metadata'].get('llm_reason_error'):
                assert data['warnings'], 'Reason fallback must be visible to the UI'
            assert calls['search_'+mode.lower()] == before['search_'+mode.lower()]+1
            if mode != 'IMAGE': assert calls['encode_query'] > before['encode_query']
            if mode != 'TEXT': assert calls['encode_image'] > before['encode_image']
            if mode != 'IMAGE':
                expected_parser = 'ollama' if args.expect == 'READY' else 'fallback'
                assert data['metadata']['parser']['parser_mode'] == expected_parser
            if mode == 'MULTIMODAL': assert data['metadata']['alpha'] == 0.8
            if mode == 'IMAGE': assert data['results'][0]['product_id'] == '271312499'
            for item in data['results']:
                original = by_id[item['product_id']]
                for field in ('price_vnd','rating','product_name','category','review_evidence','image_url'):
                    assert item[field] == original[field], field
                assert item['final_score'] is not None
                assert item['text_score'] is not None if mode != 'IMAGE' else item['text_score'] is None
                assert item['image_score'] is not None if mode != 'TEXT' else item['image_score'] is None
            # Compare the complete ranking/scores with a direct backend call.
            image = api.validate_uploaded_image(image_bytes, filename='271312499.jpg')
            if mode == 'TEXT': direct = service.search_text(query, top_k=3)
            elif mode == 'IMAGE': direct = service.search_image(image, top_k=3)
            else: direct, _ = service.search_multimodal(query, image, top_k=3)
            image.close()
            assert data['results'] == api._result_records(direct), 'HTTP adapter changed backend results'
            report['responses'][mode] = {'elapsed_ms':elapsed, 'payload': data}
            report['checks'][mode+'_real_inference_exact_backend_parity'] = True

        bounded = post({'query':'tai nghe bluetooth','category':category,'min_price':100000,'max_price':1000000,'top_k':5},image_bytes)
        bounded.raise_for_status()
        assert bounded.json()['results'] and all(100000 <= p['price_vnd'] <= 1000000 and p['category'] == category for p in bounded.json()['results'])
        report['responses']['multimodal_filters'] = bounded.json()
        report['checks']['multimodal_price_category'] = True
        empty = post({'query':'tai nghe bluetooth','max_price':1})
        assert empty.status_code == 200 and empty.json()['results'] == []
        invalid_cases = [({'query':''},None,'a.jpg',400), ({'query':'x','top_k':0},None,'a.jpg',400),
                         ({'query':'x','category':'nonexistent'},None,'a.jpg',400),
                         ({'query':''},b'not an image','a.jpg',400),
                         ({'query':''},image_bytes,'a.exe',400),
                         ({'query':''},b'x'*(api.MAX_UPLOAD_BYTES+1),'a.jpg',400),
                         ({'query':'x','min_price':100,'max_price':1},None,'a.jpg',400)]
        for fields, content, filename, expected in invalid_cases:
            bad = post(fields,content,filename)
            assert bad.status_code == expected and bad.json()['status'] == 'error'
            assert 'D:\\' not in bad.text
        for path in ['/data/images/%2e%2e%5csrc%5cconfig.py', '/data/images/%2e%2e%2fsrc%2fconfig.py', '/src/config.py','/data/images/missing.jpg']:
            bad = requests.get(base+path,timeout=10)
            assert bad.status_code in (400,404)
        verify_unsupported_media_type()
        report['checks']['empty_invalid_upload_bounds_traversal'] = True
        report['calls'] = dict(calls)
        if args.expect == 'READY': assert calls['_chat_json'] > 0
        else: assert calls['_chat_json'] == 0
        report['ok'] = True
    except Exception as error:
        report['ok'] = False
        report['error'] = repr(error)
        raise
    finally:
        (output/f'api_{args.expect.lower()}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k != 'responses'}, ensure_ascii=False, indent=2))
        server.shutdown()
        server.server_close()

if __name__ == '__main__': main()
