# ydy8989.github.io — Jekyll → Astro 이관 프로젝트

이 문서는 Claude Code 세션 간 인계용이다. 이 레포에서 작업을 이어갈 때 먼저 읽는다.

## 지금 무엇을 하는 중인가

`ydy8989.github.io`를 **Jekyll → Astro로 이관**하고, **정체성을 "ML 블로그"에서 "포트폴리오·이력서 사이트"로 전환**하는 작업.

### 🔀 방향 전환 (중요, 사용자 결정)
- 블로그 포스팅을 사실상 안 하게 됐다(글 37편 전부 2020~2022, 이후 없음). 그래서 **블로그를 앞문에서 내리고, 포트폴리오/이력서를 중심**으로 재편했다.
- **홈(`/`) = 프로젝트 커버플로우(coverflow)** — 시간순 프로젝트 카드를 OTT/원통 느낌으로 좌우 회전+페이드. `src/components/ProjectCoverflow.astro` + `src/data/projects.ts`.
- **블로그 37편은 삭제하지 않고 아카이브로 강등** — `src/pages/archive.astro`(`/archive/`), **네비에서 완전히 숨김**. URL·Disqus·SEO 보존이 목적이니 절대 지우지 말 것.
- 네비는 이제 **About 하나만** 노출. `/category/*` 페이지는 빌드되지만 링크 안 함.

### 왜 옮겼는가 (원래 이관 동기)
- 기존 Jekyll 사이트는 beautiful-jekyll 테마를 **컴파일된 채로 통째 복사(vendoring)**해 쓴다.
  `_sass`/SCSS 원본이 없고 `assets/css/`에 압축된 CSS만 있어서 **UI를 바꾸기가 거의 불가능**하다.
- 여러 블로그(`yooniversAI.github.io`, `do-yooniverse.github.io`)로 흩어졌던 것을
  주 계정 `ydy8989`로 다시 모으는 것이 목적.

### 스택
- Astro 7.x + Tailwind 4.x(`@tailwindcss/vite`) + `@astrojs/mdx`, `@astrojs/sitemap`
- 수식: `remark-math` + `rehype-katex`
- kramdown IAL: 자체 플러그인 `src/plugins/remark-ial.mjs`
- Node 24.15.0, TypeScript strict
- 개발/빌드: `npm run dev`(4321), `npm run build`, `npm run preview`

## 브랜치 상태

- 작업 브랜치: **`astro-migration`** (원격 `origin/astro-migration`과 동기화됨)
- `master` = **라이브 Jekyll 사이트, 절대 건드리지 않음** (배포 전환 전까지)
- Jekyll 원본(`_posts/`, `_config.yml`, `_layouts/` 등)은 아직 이 브랜치에 남아 있다 — 이관 소스이자 참고용. 배포 전환 후 정리 예정.

## 완료된 것 (검증까지 끝남)

**37편 전량 이관.** 전 항목 통과:

| 항목 | 결과 |
|---|---|
| permalink `/YYYY-MM-DD-slug/` | 라이브 사이트와 대조해 200 확인 (URL 정확 재현) |
| KaTeX 수식 | 에러 0 (아래 "고친 것" 참고) |
| kramdown IAL | 잔존 0 |
| 로컬 이미지 448개 참조 | 깨진 링크 0 |
| 외부 이미지 | 412장 다운로드 → `public/assets/posts/` |
| Disqus | 누락 0, 식별자 = 원래 절대 URL |

### 이관 방식
- 카테고리 하위폴더(NLP/REVIEW/boostcamp…)의 글을 평평하게 `src/content/blog/`로. 폴더는 URL에 영향 없으므로 `categories` frontmatter로 보존.
- 순수 마크다운으로 변환(Liquid 태그는 원래 0개라 걷어낼 것 없었음).
- 외부 이미지는 회수해 로컬로 교체, 로컬 상대경로(`../../assets/img/...`)는 `/assets/img/...`로 표준화.
- 재현용 스크립트: **`scripts/migrate_ydy.py`** (전량 재실행하려면 `ONLY` 필터가 비어 있는지 확인).

## 핵심 파일

- `astro.config.mjs` — `site`, **`trailingSlash: 'always'`**(URL/Disqus 일치 필수), 플러그인 등록
- `src/content.config.ts` — blog 스키마: `title, description, pubDate, slug, categories, tags, disqusId`
- `src/pages/[...permalink].astro` — **글 상세 라우트**. `slug`를 permalink로. 목차·프로즈 CSS·수식 CSS·IAL 이미지 CSS·Disqus 포함
- `src/pages/index.astro` — 글 목록 (현재는 최소 구성)
- `src/components/Disqus.astro` — 댓글. shortname `ydy8989-github-io`, identifier=`disqusId`, 뷰포트 진입 시 지연 로드
- `src/plugins/remark-ial.mjs` — kramdown IAL(`{: .center}`, `{: width="80%"}`) → 클래스(`center`, `w-xx`)
- `src/components/Header.astro`, `Footer.astro`, `layouts/Base.astro`
- `src/styles/global.css` — 테마 토큰(색·폰트). **현재 따뜻한 "디지털 가든" 팔레트가 들어가 있는데, 디자인 방향 미확정이라 재검토 대상**

## 이관 중 고친 것 (원본 글의 문제였음, 이관 버그 아님)

1. **`2021roberta.md`** — 날짜 접두사가 없어 Jekyll이 무시하던 초안(라이브 404). git 최초 커밋일 기준 `/2021-07-21-roberta/` 부여. 발행된 적 없어 URL·댓글 충돌 없음. → **실제 발행할지 결정 필요**
2. **`2021-02-17-attention.md`** — `\text{length_of_prediction}` 등 `\text{}` 안 언더스코어. MathJax는 관대했지만 KaTeX는 엄격. 라벨이라 공백으로, `precision_i`만 아래첨자로 수정.
3. **`2021-02-05-gan2.md`** — 리스트 항목 안에 탭 들여쓴 여러 줄 `$$` align 블록이 마크다운 처리 중 쪼개짐. 최상위 display 블록 + `aligned` + 중괄호 첨자로 정리.
4. **죽은 외부 이미지 2개**는 원본 URL로 남김 (라이브에서도 이미 깨져 있던 것): vox-cdn 썸네일(400), brianzhang01 `.mp4`(영상 임베드).

## 함정 (다시 헤매지 말 것)

- **Astro 콘텐츠 캐시는 `node_modules/.astro/data-store.json`에 있다** (`.astro/` 아님).
  마크다운 파이프라인(플러그인 등)을 고쳤는데 빌드에 반영이 안 되면 이 파일을 지워라:
  `rm -rf node_modules/.astro .astro dist && npm run build`
- **Disqus 식별자는 원래 절대 URL과 반드시 일치**해야 스레드가 안 끊긴다. `disqusId`를 함부로 바꾸지 말 것.
- **`trailingSlash: 'always'`** 없으면 permalink/Disqus가 어긋난다.
- `astro dev stop`이 프로세스를 항상 죽이지 못해 서버가 4322로 밀릴 수 있다. 포트 정리 후 재기동.
- 이미지 다운로드 시 `.mp4` 등 비이미지 링크는 회수 대상이 아님.

## 완료된 것 (2차 세션 — 네비/누락 페이지)

- **About 페이지** (`src/pages/about.astro` + `src/data/profile.ts`) — `do-yooniverse.github.io`의 About를 이식. 최신 커리어(라이앤캐처스 2021.11~현재)를 담은 `profile.ts` 기반 + `aboutme.md`의 학력(광운대 수학 학·석사) Education 섹션 추가. Resume/CV PDF 다운로드 링크 포함.
- **이력서/CV PDF** — `port_cv/`의 PDF 2개를 `public/port_cv/`로 복사해 옛 URL(`/port_cv/resume/...`, `/port_cv/cv/...`) 그대로 유지. **결정: 포트폴리오 페이지는 안 만들고 PDF만 살림.**
- **카테고리 페이지** — `src/pages/category/index.astro`(전체 목록+글수) + `[category].astro`. 옛 Jekyll URL 소문자(`/category/nlp/`, `/boostcamp/`, `/gnn/`, `/etc/`) 재현. frontmatter categories는 대문자라 `.toLowerCase()`로 매칭.
- **Header 네비** — `Posts / Categories / About` + 홈만 exact 매칭. `Base.astro` 콘솔 인사말의 `do-yooniverse` 브랜딩을 `ydy8989`로 정리.
- 빌드 45페이지 정상.

## 남은 일 (다음 세션 우선순위)

1. **배포 전환** — 현재 라이브는 `master`의 Jekyll. GitHub Actions로 Astro 빌드 → Pages 배포로 전환해야 함. (`.github/workflows` 없음)
2. **Jekyll 잔재 정리** — 배포 전환 후 `_posts`, `_layouts`, `_includes`, `_config.yml`, `Gemfile*`, `dest/`, `feed.xml`, `sitemap.xml`(Astro가 생성), 컴파일된 `assets/css` 등 제거.
3. **디자인** — 사용자는 위트 있고 fancy/유머러스한 UI를 원함(레퍼런스: Josh Comeau, samwho, Maggie Appleton 등). 현재는 검증용 최소 뼈대.
   참고: `do-yooniverse.github.io` 레포에 어텐션-호버 글목록(`lib/attention.ts`, `PostList.astro`), 디지털 가든(완성도 라벨 `StageBadge`), 프로젝트 카드(`ProjectCard`) 등을 만들어 뒀다. 필요하면 이식 가능.
4. **부가**: Search(옛 `/search/`), 태그 페이지(옛 `tags.html`), RSS, OG 이미지.
5. **알려진 이슈**: `Base.astro`가 `/favicon.svg`를 참조하는데 `public/`에 파일이 없어 깨진 링크. favicon 추가 필요.

**About 내용 주의**: `aboutme.md`(2021에서 멈춤)가 아니라 `do-yooniverse`의 `profile.ts`가 최신 소스. 2021 이후 이력(라이앤캐처스 재직 중) 갱신은 사용자 확인 필요.

## 관련 레포

- `do-yooniverse.github.io` (이웃 폴더) — 원래 이 이관을 실험하던 곳. UI 컴포넌트 실험체가 남아 있음.
- `yooniversAI.github.io` (`~/Documents/코드아카이브/` 및 `~/Documents/code/2025/blogs/`) — 중간에 거쳐간 블로그. 6편 중 일부는 ydy8989 글의 재탕이었음.
