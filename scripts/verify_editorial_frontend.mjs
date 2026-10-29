// JavaScript boundary checks against the real HTTP API. This is NOT a browser/manual test.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {File} from 'node:buffer';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = fs.readFileSync(path.join(root, 'hopgu_v2_editorial_frontend.html'), 'utf8');
const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].map(m => m[1]).join('\n');
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {value:'', innerHTML:'', textContent:'', disabled:false,
    classList:{add(){},remove(){},toggle(){}}, firstElementChild:{textContent:''},
    querySelectorAll(){return [];},setAttribute(){},addEventListener(){},style:{}});
  return elements.get(id);
}
const requests = [];
let unavailable = false;
const context = vm.createContext({console, URL, FormData, File, performance,
  localStorage:{getItem(){return null;},setItem(){}}, tailwind:{},
  window:{addEventListener(){},scrollTo(){}},
  document:{getElementById:element,querySelectorAll(){return [];}},
  fetch:async (route, options) => {
    const response = await fetch((unavailable ? 'http://127.0.0.1:1' : 'http://127.0.0.1:8000')+route, options);
    if (route === '/api/recommend') {
      const fields = [...options.body.entries()].map(([key,value]) => ({key,
        value: typeof value === 'string' ? value : {name:value.name,size:value.size,type:value.type}}));
      requests.push({route, fields, status:response.status, payload:await response.clone().json()});
    }
    return response;
  }
});
vm.runInContext(scripts, context);
const run = (code) => vm.runInContext(code, context);
const checks = {};
element('searchTopKSelect').value='3';
element('searchMaxPriceRange').value='0';
await run('trySyncWithFullBackend()');
assert.equal(run('CATALOG_DATA.length'),860);
checks.catalog_api = true;
const target = run('findProduct("271312499")');
const imageResponse = await fetch('http://127.0.0.1:8000'+target.image_url);
const bytes = await imageResponse.arrayBuffer();
context.testImageFile = new File([bytes], '271312499.jpg', {type:'image/jpeg'});
for (const mode of ['TEXT','IMAGE','MULTIMODAL']) {
  element('aiSearchInput').value=mode==='IMAGE'?'':'tai nghe bluetooth';
  if (mode!=='TEXT') await run('selectReference(testImageFile)');
  const pending = run('runAiSearch()');
  assert.equal(element('runAiSearchBtn').disabled,true);
  assert.match(element('searchResultsGrid').innerHTML,/Đang tìm sản phẩm hợp gu/);
  await pending;
  const last = requests.at(-1);
  assert.equal(last.payload.mode,mode);
  assert.equal(last.status,200);
  assert.equal(run('RECOMMENDATION_DATA.length'),3);
  assert.equal(element('runAiSearchBtn').disabled,false);
  assert.match(element('parserSummaryText').textContent,new RegExp(mode));
  for (const warning of last.payload.warnings || []) assert.ok(element('parserSummaryText').textContent.includes(warning));
  const grid = element('searchResultsGrid').innerHTML;
  for (const product of last.payload.results) {
    assert.ok(grid.includes(String(product.final_score)));
    assert.ok(grid.includes(product.image_url));
  }
  if (mode!=='TEXT') assert.equal(last.fields.find(f=>f.key==='image').value.size,bytes.byteLength);
  assert.ok(!last.fields.some(f=>/score|alpha/.test(f.key)));
  checks[mode+'_formdata_render'] = true;
}
const product = requests.at(-1).payload.results[0];
run(`openProductModal(${JSON.stringify(product.product_id)})`);
assert.ok(element('productModalContent').innerHTML.includes(run(`escapeHtml(${JSON.stringify(product.reason)})`)));
assert.equal(run('escapeHtml("<img src=x onerror=alert(1)>")'),'&lt;img src=x onerror=alert(1)&gt;');
checks.modal_evidence_and_escape = true;
element('searchMaxPriceRange').value='1';
await run('runAiSearch()');
assert.equal(run('RECOMMENDATION_DATA.length'),0);
assert.match(element('searchResultsGrid').innerHTML,/Không tìm thấy sản phẩm/);
checks.empty_no_catalog_fallback = true;
unavailable = true;
await run('runAiSearch()');
assert.equal(run('RECOMMENDATION_DATA.length'),0);
assert.equal(run('backendState'),'ERROR');
assert.match(element('searchResultsGrid').firstElementChild.textContent,/Không thể kết nối hệ thống gợi ý/);
checks.connection_error_no_fake_results = true;
const report={ok:true,browser_manual:false,checks,requests};
fs.writeFileSync(path.join(root,'integration_verification','frontend_api.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify({ok:true,browser_manual:false,checks}));
