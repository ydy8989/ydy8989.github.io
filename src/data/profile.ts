// 이력·프로필 데이터. 표현(HTML)은 절대 넣지 않는다 — 컴포넌트가 담당.
export const profile = {
  name: '윤도연',
  nameEn: 'Doyeon Yoon',
  role: 'Data Scientist',
  tagline: 'NLP와 LLM으로 쓸모 있는 제품을 만듭니다.',
  email: 'ydy89899@gmail.com',
  resumeUrl: '/port_cv/resume/Doyeon_Yoon_resume.pdf',
  cvUrl: '/port_cv/cv/Doyeon_Yoon_CV.pdf',
  social: [
    { label: 'GitHub', href: 'https://github.com/ydy8989' },
  ],
} as const;

// 문자열은 단일 항목, 객체는 하위 들여쓰기 항목을 가진 항목.
export type Bullet = string | { text: string; children: string[] };

export type CareerEntry = {
  role: string;
  org: string;
  location: string;
  from: string;
  to: string | null; // null = 재직 중
  bullets: Bullet[];
};

export const career: CareerEntry[] = [
  {
    role: '데이터 사이언티스트',
    org: '컬리',
    location: '서울',
    from: '2025.04',
    to: null,
    bullets: [
      '컬리몰 검색 서비스 내 검색/추천 배치 운영',
      '일부 검색어를 대상으로 Semantic Search PoC를 수행하여 검색 품질 개선 가능성 검증',
      'Semantic Search를 전체 검색에 적용하고 Hybrid Search 체계로 고도화',
      '검색 개편 1차: Unknown Query 대응 강화를 위한 시맨틱 검색 모델 개선 및 검색 품질 향상',
      '검색 개편 2차: IRF 및 최신성 기반 랭킹 피쳐를 도입하여 검색 결과 정렬 로직 고도화',
    ],
  },
  {
    role: 'Data Scientist & ML Engineer',
    org: '라이앤캐처스',
    location: '서울',
    from: '2021.11',
    to: '2025.03',
    bullets: [
      '글로컬 LLM RAG 서버 구축',
      '와인나라 추천시스템 ML 모델링',
      '사내 추천시스템 개발',
      'AI 채용 자동화(표절검사기)',
      '대스타 해결사 플랫폼 준우승',
    ],
  },
  {
    role: 'Data Scientist',
    org: 'ScatterX',
    location: '',
    from: '2019.06',
    to: '2020.03',
    bullets: [
      '삼성전자 반도체 공정 이상탐지 모델 개발 (전 공정 적용)',
      'Autoencoder·SGAN·회귀 병렬 구조로 인터락 7종 케이스 분류 (Precision 0.99)',
      '전 공정 적용으로 휴먼 리소스 80% 절감',
    ],
  },
  {
    role: '연구 인턴',
    org: '포항공대 인공지능연구원(PIAI)',
    location: '포항',
    from: '2018.11',
    to: '2018.12',
    bullets: ['3D U-net 기반 의료영상 segmentation', 'Semi-supervised learning 연구'],
  },
];

export type EducationEntry = {
  degree: string;
  org: string;
  location: string;
  year: string;
  note?: string;
  noteHref?: string;
};

export const education: EducationEntry[] = [
  {
    degree: '수학과 석사 (M.S.)',
    org: '광운대학교',
    location: '서울',
    year: '2017',
    note: '조합론 · 그래프 anti-bandwidth (학위논문)',
    noteHref: 'http://www.riss.kr/link?id=T14494628',
  },
  {
    degree: '수학과 학사 (B.S.)',
    org: '광운대학교',
    location: '서울',
    year: '2015',
  },
  {
    degree: '기계시스템·디자인공학과',
    org: '서울과학기술대학교',
    location: '서울',
    year: '2013',
  },
];

export const skills = {
  '언어 · 프레임워크': ['Python', 'PyTorch', 'FastAPI', 'LangChain'],
  'ML · NLP': ['LLM', 'RAG', 'Sentence Embedding'],
  '인프라': ['Docker', 'Milvus', 'Redis', 'AWS', 'OpenSearch', 'Kubeflow', 'Airflow'],
} as const;
