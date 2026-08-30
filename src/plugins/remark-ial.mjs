import { visit } from 'unist-util-visit';

/**
 * kramdown IAL(Inline Attribute List) 지원.
 *
 * Jekyll 시절 글에 `![img](url){: width="50%"}{: .center}` 형태로 적힌 문법을
 * 그대로 살린다. kramdown 전용 확장이라 표준 remark는 이걸 모르고 본문에
 * 글자 그대로 출력해버린다.
 *
 * 지원하는 표기 (원문에 실제로 섞여 있는 변형 모두 포함):
 *   {: .center}  {:.center}
 *   {: width="80%"}  {:width="80%"}  {:.width="80%"}   ← 마지막은 원문 오타
 */

// 이미지 바로 뒤에 붙은 하나 이상의 {: ... } 덩어리
const IAL_RUN = /^\s*(?:\{:[^}]*\}\s*)+/;

// .클래스 — `.width="80%"` 같은 속성 오타를 클래스로 오인하면 안 된다.
// (?![\w-]) 가 없으면 정규식이 되돌아가며 `.widt` 를 클래스로 잡아낸다.
const CLASS_RE = /\.([A-Za-z_-][\w-]*)(?![\w-]|\s*=)/g;

// 따옴표는 곧은 것과 둥근 것을 모두 받는다. Astro 는 smartypants 가 기본
// 활성이라 이 플러그인이 보기 전에 "80%" 가 “80%” 로 바뀌어 있다.
const WIDTH_RE = /width\s*=\s*["'“”‘’]?\s*([\d.]+)\s*%?\s*["'“”‘’]?/i;

export function remarkIal() {
  return (tree) => {
    visit(tree, 'paragraph', (para) => {
      for (let i = 0; i < para.children.length - 1; i++) {
        const img = para.children[i];
        const next = para.children[i + 1];
        if (img.type !== 'image' || next.type !== 'text') continue;

        const match = next.value.match(IAL_RUN);
        if (!match) continue;
        const raw = match[0];

        const classes = [...raw.matchAll(CLASS_RE)].map((m) => m[1]);
        const width = raw.match(WIDTH_RE)?.[1];

        // 폭도 클래스로 넘긴다. 인라인 style 은 Astro 가 이미지 노드를
        // 재구성하면서 떨어뜨리기 때문에(클래스는 유지됨) 살아남지 못한다.
        // 5% 단위로 반올림해 w-xx 클래스에 대응시킨다.
        if (width) {
          const pct = Math.min(100, Math.max(5, Math.round(parseFloat(width) / 5) * 5));
          classes.push(`w-${pct}`);
        }

        if (classes.length) {
          const data = (img.data ??= {});
          const props = (data.hProperties ??= {});
          props.className = [...(props.className ?? []), ...classes];
        }

        // 소비한 IAL 텍스트는 본문에서 제거
        next.value = next.value.slice(raw.length);
        if (!next.value.trim()) {
          para.children.splice(i + 1, 1);
        }
      }
    });
  };
}

export default remarkIal;
