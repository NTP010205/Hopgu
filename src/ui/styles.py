APP_CSS = """
<style>
  :root {
    --hg-accent: #B23A52;
    --hg-accent-dark: #8F2940;
    --hg-ink: #1C2433;
    --hg-muted: #5F6877;
    --hg-canvas: #F8E4DA;
    --hg-canvas-soft: #FFF8F4;
    --hg-surface: rgba(255, 255, 255, .86);
    --hg-line: rgba(106, 85, 78, .16);
    --hg-success: #18794E;
    --hg-warning: #A15C13;
    --hg-error: #B4233A;
    --hg-shadow: 0 22px 55px -34px rgba(57, 39, 45, .42);
  }

  html {scroll-behavior: smooth;}
  .stApp {
    color: var(--hg-ink);
    background:
      radial-gradient(circle at 9% 3%, rgba(255,255,255,.95) 0, rgba(255,255,255,0) 31rem),
      radial-gradient(circle at 92% 18%, rgba(206,151,133,.22) 0, rgba(206,151,133,0) 28rem),
      linear-gradient(150deg, var(--hg-canvas-soft) 0%, var(--hg-canvas) 52%, #F2D8CF 100%);
    font-family: "Be Vietnam Pro", "Segoe UI", Arial, sans-serif;
  }
  .stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: .24;
    background-image: radial-gradient(rgba(96,68,70,.13) .6px, transparent .6px);
    background-size: 7px 7px;
  }
  .block-container {max-width: 1260px; padding-top: 2.25rem; padding-bottom: 4rem;}
  h1, h2, h3, p {color: var(--hg-ink);}
  h1, h2, h3, .hopgu-display {
    font-family: Georgia, "Times New Roman", serif;
    letter-spacing: -.025em;
  }
  [data-testid="stSidebar"] {
    background: rgba(255, 250, 247, .9);
    border-right: 1px solid var(--hg-line);
  }
  [data-testid="stSidebarNav"] a {border-radius: 999px; margin: .18rem .65rem;}
  [data-testid="stSidebarNav"] a[aria-current="page"] {
    background: rgba(178, 58, 82, .1);
    color: var(--hg-accent-dark);
    font-weight: 700;
  }

  .hopgu-hero {
    position: relative;
    overflow: hidden;
    padding: clamp(2rem, 5vw, 4.25rem);
    border: 1px solid rgba(255,255,255,.78);
    border-radius: 32px;
    background:
      radial-gradient(circle at 85% 18%, rgba(255,255,255,.92) 0 3rem, rgba(255,255,255,0) 3.1rem),
      linear-gradient(125deg, rgba(255,255,255,.9), rgba(255,241,234,.78));
    box-shadow: var(--hg-shadow), inset 0 1px 0 rgba(255,255,255,.9);
    backdrop-filter: blur(18px) saturate(1.08);
    margin-bottom: 1.75rem;
  }
  .hopgu-hero::after {
    content: "";
    position: absolute;
    width: 18rem;
    height: 18rem;
    right: -7rem;
    bottom: -9rem;
    border: 1px solid rgba(178,58,82,.14);
    border-radius: 50%;
    box-shadow: 0 0 0 3rem rgba(178,58,82,.035), 0 0 0 6rem rgba(178,58,82,.02);
  }
  .hopgu-hero__content {position: relative; z-index: 1; max-width: 780px;}
  .hopgu-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    margin-bottom: 1rem;
    color: var(--hg-accent-dark);
    font-size: .75rem;
    font-weight: 800;
    letter-spacing: .13em;
    text-transform: uppercase;
  }
  .hopgu-eyebrow::before {content: ""; width: 1.6rem; height: 1px; background: currentColor;}
  .hopgu-hero h1 {
    max-width: 720px;
    margin: 0 0 .9rem;
    color: var(--hg-ink);
    font-size: clamp(2.35rem, 5vw, 4.6rem);
    font-weight: 500;
    line-height: 1.03;
  }
  .hopgu-hero p {
    max-width: 670px;
    margin: 0;
    color: var(--hg-muted);
    font-size: clamp(.98rem, 1.7vw, 1.12rem);
    line-height: 1.72;
  }

  .hopgu-section-heading {margin: 2rem 0 1rem;}
  .hopgu-section-heading h2 {margin: 0; font-size: clamp(1.65rem, 2.6vw, 2.25rem); font-weight: 500;}
  .hopgu-section-heading p {margin: .35rem 0 0; color: var(--hg-muted); font-size: .92rem;}
  .hopgu-kicker {color: var(--hg-accent-dark); font-size: .72rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase;}
  .hopgu-muted {color: var(--hg-muted);}

  .hopgu-badge-row {display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: .7rem;}
  .hopgu-badge {
    display: inline-flex;
    align-items: center;
    gap: .3rem;
    padding: .3rem .62rem;
    border: 1px solid rgba(178,58,82,.16);
    border-radius: 999px;
    background: rgba(178,58,82,.08);
    color: var(--hg-accent-dark);
    font-size: .67rem;
    font-weight: 800;
    letter-spacing: .07em;
  }
  .hopgu-badge--fallback {border-color: rgba(161,92,19,.2); background: #FFF5E8; color: #8A4B0D;}
  .hopgu-badge--catalog {border-color: rgba(28,36,51,.13); background: rgba(28,36,51,.055); color: var(--hg-ink);}
  .hopgu-category {
    display: inline-block;
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--hg-muted);
    font-size: .73rem;
    font-weight: 650;
    letter-spacing: .025em;
  }
  .hopgu-product-title {
    height: 2.88em;
    margin: .42rem 0 .38rem;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    color: var(--hg-ink);
    font-family: Georgia, "Times New Roman", serif;
    font-size: 1.02rem;
    font-weight: 650;
    line-height: 1.44;
  }
  .hopgu-price {color: var(--hg-accent-dark); font-size: 1.16rem; font-weight: 850; letter-spacing: -.02em;}
  .hopgu-meta {color: var(--hg-muted); font-size: .79rem; margin: .35rem 0 .25rem; line-height: 1.45;}
  .hopgu-reason {
    padding: .7rem .8rem;
    border-left: 3px solid var(--hg-accent);
    border-radius: 0 12px 12px 0;
    background: rgba(255, 247, 243, .92);
    color: #4B3440;
    font-size: .84rem;
    line-height: 1.58;
  }
  .hopgu-score-row {display: grid; grid-template-columns: 1fr auto; gap: .65rem; padding: .28rem 0; color: var(--hg-muted); font-size: .82rem;}
  .hopgu-score-row strong {color: var(--hg-ink); font-variant-numeric: tabular-nums;}

  div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(255,255,255,.76);
    border-radius: 22px;
    background: var(--hg-surface);
    box-shadow: var(--hg-shadow), inset 0 1px 0 rgba(255,255,255,.75);
    backdrop-filter: blur(16px) saturate(1.08);
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
  }
  div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(255,255,255,.95);
    box-shadow: 0 26px 62px -38px rgba(57,39,45,.55), inset 0 1px 0 #fff;
  }
  div[data-testid="stImage"] img {
    width: 100%;
    aspect-ratio: 1 / 1;
    object-fit: contain;
    border-radius: 15px;
    background: linear-gradient(145deg, #fff, #F8F3F0);
    border: 1px solid rgba(86,71,66,.08);
  }
  details[data-testid="stExpander"] {border-color: var(--hg-line); border-radius: 13px; background: rgba(255,255,255,.5);}
  details[data-testid="stExpander"] summary {font-size: .82rem; font-weight: 700; color: var(--hg-ink);}

  .hopgu-stat-grid {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin: 1rem 0 1.75rem;}
  .hopgu-stat-card {
    min-height: 128px;
    padding: 1.25rem;
    border: 1px solid rgba(255,255,255,.76);
    border-radius: 22px;
    background: var(--hg-surface);
    box-shadow: var(--hg-shadow);
  }
  .hopgu-stat-label {color: var(--hg-muted); font-size: .76rem; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;}
  .hopgu-stat-value {margin: .42rem 0 .28rem; color: var(--hg-ink); font-family: Georgia, "Times New Roman", serif; font-size: clamp(1.75rem, 3vw, 2.5rem); line-height: 1;}
  .hopgu-stat-detail {color: var(--hg-muted); font-size: .76rem; line-height: 1.45;}

  .hopgu-eval-title {
    margin-bottom: .85rem;
    color: var(--hg-ink);
    font-family: Georgia, "Times New Roman", serif;
    font-size: 1.08rem;
    font-weight: 700;
    white-space: nowrap;
  }
  .hopgu-eval-grid {display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .55rem;}
  .hopgu-eval-metric {padding: .7rem; border: 1px solid var(--hg-line); border-radius: 13px; background: rgba(255,255,255,.58);}
  .hopgu-eval-label {color: var(--hg-muted); font-size: .64rem; font-weight: 750; line-height: 1.3;}
  .hopgu-eval-value {margin-top: .25rem; color: var(--hg-ink); font-family: Georgia, "Times New Roman", serif; font-size: 1.02rem; font-weight: 700; font-variant-numeric: tabular-nums;}
  .hopgu-eval-latency {margin-top: .75rem; color: var(--hg-muted); font-size: .72rem;}

  .hopgu-status-card {
    position: relative;
    min-height: 142px;
    padding: 1.25rem 1.25rem 1.1rem;
    border: 1px solid var(--hg-line);
    border-radius: 20px;
    background: rgba(255,255,255,.74);
    box-shadow: var(--hg-shadow);
  }
  .hopgu-status-card::before {content: ""; position: absolute; inset: 0 auto 0 0; width: 4px; border-radius: 20px 0 0 20px; background: var(--hg-muted);}
  .hopgu-status-card--ready::before {background: var(--hg-success);}
  .hopgu-status-card--fallback::before, .hopgu-status-card--pending::before {background: var(--hg-warning);}
  .hopgu-status-card--error::before {background: var(--hg-error);}
  .hopgu-status-top {display: flex; align-items: center; justify-content: space-between; gap: .75rem;}
  .hopgu-status-label {font-family: Georgia, "Times New Roman", serif; font-size: 1.08rem; font-weight: 650;}
  .hopgu-status-pill {padding: .26rem .55rem; border-radius: 999px; font-size: .65rem; font-weight: 850; letter-spacing: .08em;}
  .hopgu-status-pill--ready {background: #EAF7F0; color: var(--hg-success);}
  .hopgu-status-pill--fallback, .hopgu-status-pill--pending {background: #FFF4E3; color: var(--hg-warning);}
  .hopgu-status-pill--error {background: #FDECEF; color: var(--hg-error);}
  .hopgu-status-detail {margin-top: .75rem; color: var(--hg-muted); font-size: .8rem; line-height: 1.5; overflow-wrap: anywhere;}

  .hopgu-state {
    max-width: 680px;
    margin: 1.2rem auto;
    padding: 2.2rem 1.5rem;
    border: 1px dashed rgba(100,78,73,.25);
    border-radius: 24px;
    background: rgba(255,255,255,.58);
    text-align: center;
  }
  .hopgu-state__icon {display: grid; place-items: center; width: 3rem; height: 3rem; margin: 0 auto .8rem; border-radius: 15px; background: rgba(178,58,82,.09); font-size: 1.35rem;}
  .hopgu-state h3 {margin: 0 0 .4rem; font-size: 1.25rem;}
  .hopgu-state p {margin: 0; color: var(--hg-muted); font-size: .88rem; line-height: 1.6;}

  .hopgu-step {padding: 1.15rem 1.2rem; border-top: 1px solid var(--hg-line);}
  .hopgu-step strong {display: block; margin-bottom: .2rem; font-family: Georgia, "Times New Roman", serif; font-size: 1rem; color: var(--hg-ink);}
  .hopgu-step span {color: var(--hg-muted); font-size: .8rem; line-height: 1.45;}

  .stButton > button {
    min-height: 2.75rem;
    border-radius: 999px;
    border-color: rgba(28,36,51,.18);
    font-weight: 750;
    transition: transform .16s ease, box-shadow .16s ease, background .16s ease;
  }
  .stButton > button:hover {transform: translateY(-1px); border-color: rgba(28,36,51,.3);}
  .stButton > button[kind="primary"] {
    border-color: var(--hg-ink);
    background: var(--hg-ink);
    color: #fff;
    box-shadow: 0 10px 24px -14px rgba(28,36,51,.75);
  }
  .stButton > button[kind="primary"] p,
  .stButton > button[kind="primary"] span {color: #fff !important;}
  .stButton > button[kind="primary"]:hover {border-color: #303B4D; background: #303B4D;}
  div[data-baseweb="select"] > div,
  div[data-testid="stTextInputRootElement"],
  div[data-testid="stTextAreaRootElement"],
  div[data-testid="stFileUploaderDropzone"] {
    border-color: rgba(88,70,66,.19);
    border-radius: 16px;
    background: rgba(255,255,255,.68);
  }
  div[data-testid="stTextAreaRootElement"] textarea {font-size: 1.02rem; line-height: 1.55;}
  div[data-testid="stFileUploaderDropzone"] {padding: 1rem;}
  [data-testid="stMetric"] {
    padding: 1rem;
    border: 1px solid rgba(255,255,255,.76);
    border-radius: 18px;
    background: rgba(255,255,255,.72);
    box-shadow: var(--hg-shadow);
  }
  [data-testid="stMetricValue"] {font-family: Georgia, "Times New Roman", serif; color: var(--hg-ink);}
  [data-testid="stDataFrame"] {border: 1px solid var(--hg-line); border-radius: 16px; overflow: hidden;}
  [data-testid="stAlert"] {border-radius: 16px;}
  hr {border-color: var(--hg-line) !important;}

  @media (max-width: 780px) {
    .block-container {padding-top: 1.25rem;}
    .hopgu-hero {padding: 2rem 1.35rem; border-radius: 24px;}
    .hopgu-stat-grid {grid-template-columns: 1fr;}
    .hopgu-stat-card {min-height: auto;}
  }
  @media (prefers-reduced-motion: reduce) {
    html {scroll-behavior: auto;}
    *, *::before, *::after {transition: none !important;}
  }
</style>
"""
