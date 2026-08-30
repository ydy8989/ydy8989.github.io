// 홈 커버플로우 + 세로 타임라인 데이터. 최신순(위 = 가장 최근).
// 출처: 이력서(Doyeon_Yoon_resume.pdf) + 컬리 지라 이니셔티브.
// title 은 타임라인/카드에 그대로 쓰이니 짧게, 상세는 summary 로.
export type Project = {
  id: string;
  period: string;
  org: string;
  role?: string;
  title: string;
  summary: string;
  tags: string[];
  href?: string;
};

export const projects: Project[] = [
  // ── 컬리 (Data Scientist, 2025.04 ~ 재직중) ─────────────
  {
    id: 'kurly-search-reform',
    period: '2026.05 — 현재',
    org: '컬리',
    role: 'Data Scientist',
    title: '검색 개편 · 사전 고도화',
    summary:
      '형태소 분석 사전(동의어·복합어·오타)과 LLM 기반 사전 확장으로 검색 이해도를 끌어올리는 개편을 진행 중.',
    tags: ['LLM', '형태소분석', '검색품질'],
  },
  {
    id: 'kurly-irf-ranking',
    period: '2025.12 — 2026.06',
    org: '컬리',
    role: 'Data Scientist',
    title: 'IRF 기반 랭킹 피처',
    summary:
      'Implicit Relevance Feedback와 최신성 기반 피처를 도입해 검색 결과 정렬 로직을 고도화.',
    tags: ['Ranking', 'IRF', 'Feature Eng'],
  },
  {
    id: 'kurly-semantic-2',
    period: '2025.12 — 2026.02',
    org: '컬리',
    role: 'Data Scientist',
    title: 'Semantic Search 2차',
    summary:
      'SageMaker 엔드포인트 배포 파이프라인과 컷오프·운영 이슈 대응으로 시맨틱 검색을 전체 검색에 상용 적용.',
    tags: ['Semantic Search', 'SageMaker', 'MLOps'],
  },
  {
    id: 'kurly-cutoff',
    period: '2025.08 — 2025.12',
    org: '컬리',
    role: 'Data Scientist',
    title: '검색 적합도 모델링',
    summary:
      '검색어별 유사도 임계값(cut-off) 결정 모델과 스코어링 배치 파이프라인을 구성.',
    tags: ['Modeling', 'Pipeline', 'BigQuery'],
  },
  {
    id: 'kurly-semantic-poc',
    period: '2025.05 — 2025.08',
    org: '컬리',
    role: 'Data Scientist',
    title: 'Semantic Search PoC',
    summary:
      '검색어–상품 임베딩 파인튜닝으로 시맨틱 검색 도입 가능성을 검증, 상위 키워드 recall 9% 개선하여 조직 도입을 견인.',
    tags: ['PoC', 'Fine-tuning', 'Embedding'],
  },

  // ── 라이앤캐처스 (Data Scientist & ML Engineer, 2021.11 ~ 2025.03) ──
  {
    id: 'lian-hallym-rag',
    period: '2024.09 — 2025.02',
    org: '라이앤캐처스',
    role: 'ML Engineer',
    title: '한림대 RAG AI 조교',
    summary:
      '교수용 AI 조교 챗봇을 LangChain Advanced RAG로 구축. Hybrid search·IVF 인덱싱으로 생성물 저장율 20% 향상.',
    tags: ['RAG', 'LangChain', 'Hybrid Search'],
  },
  {
    id: 'lian-hr-plagiarism',
    period: '2023.06 — 2024.04',
    org: '라이앤캐처스',
    role: 'Data Scientist',
    title: 'AI 채용서류 자동평가',
    summary:
      '표절·블라인드 위반 등 14개 항목 검출 시스템. 표절 검출을 2일→2시간으로, SimCSE 도입으로 휴먼리소스 90% 절감(recall > 0.95).',
    tags: ['NLP', 'SimCSE', '표절검출'],
  },
  {
    id: 'lian-doc-ranking',
    period: '2023.01 — 2023.03',
    org: '라이앤캐처스',
    role: 'ML Engineer',
    title: '지식거래 유사문서 랭킹',
    summary:
      'KoELECTRA 임베딩 + SimCSE 리랭커로 문서 랭킹(Hit@3 = 0.92). MongoDB→FAISS 전환으로 검색 시간 66% 단축.',
    tags: ['Ranking', 'KoELECTRA', 'FAISS'],
  },
  {
    id: 'lian-daestar',
    period: '2022.09 — 2022.12',
    org: '라이앤캐처스',
    role: 'Data Scientist',
    title: '데이터 증강 대회 · 준우승',
    summary:
      '모델 정보 없이 데이터 증강 전략만으로 텍스트 분류 성능을 극대화, 대스타 해결사 플랫폼 대회 준우승(상금 9천만원).',
    tags: ['Data Augmentation', 'NLP', '준우승'],
  },

  // ── 대회 / 교육 ────────────────────────────────────
  {
    id: 'side-dst',
    period: '2021.05 — 2021.06',
    org: '사이드 프로젝트',
    title: 'DST 대회 · 1위',
    summary:
      'SOMDST 구현·최적화와 BLEU 후처리로 Dialogue State Tracking 대회 Public·Private 1위 달성.',
    tags: ['NLP', 'DST', 'SOMDST'],
  },
  {
    id: 'boostcamp',
    period: '2021.01 — 2021.06',
    org: '네이버 커넥트재단',
    title: '부스트캠프 AI Tech',
    summary:
      'NLP·CV·GNN 6개월 교육과 리더보드 대회형 프로젝트를 수행한 네이버 부스트캠프 AI Tech 1기.',
    tags: ['NLP', 'CV', 'GNN'],
    href: 'https://github.com/ydy8989/boostcamp',
  },

  // ── ScatterX (Data Scientist, 2019.06 ~ 2020.03) ───────
  {
    id: 'scatterx-anomaly',
    period: '2019.06 — 2020.03',
    org: 'ScatterX',
    role: 'Data Scientist',
    title: '반도체 센서 이상탐지',
    summary:
      '삼성전자 반도체 인터락 7종 케이스를 30일 시계열 센서로 Autoencoder·SGAN·회귀 병렬 분류(Precision 0.99). 전 공정 적용해 휴먼리소스 80% 절감.',
    tags: ['Anomaly Detection', 'SGAN', 'Time Series'],
  },

  // ── 포항공대 PIAI/PIRL (2018) ──────────────────────
  {
    id: 'piai-help',
    period: '2018.11 — 2018.12',
    org: '포항공대 PIAI',
    role: '연구 인턴',
    title: '의료영상 · HeLP 2018',
    summary:
      '아산병원 HeLP Challenge 2018에서 3D U-net 의료영상 세그멘테이션으로 최종 5위, 연구 보조·논문 서베이 수행.',
    tags: ['Segmentation', '3D U-net', 'Medical'],
    href: 'https://github.com/ydy8989/Cardiac_Segmentation',
  },
  {
    id: 'senticle',
    period: '2018.09 — 2018.10',
    org: '포항공대 PIRL',
    role: '팀 프로젝트',
    title: 'Senticle · 주가예측',
    summary:
      '뉴스로 주가 상하락을 예측하는 모델을 신뢰구간과 함께 제시하고 LIME으로 근거를 시각화한 팀 프로젝트.',
    tags: ['NLP', 'LIME', '주가예측'],
    href: 'https://github.com/ydy8989/senticle-proj',
  },
];
