# 폴더 정리 — 2026-11-18

## 변경 계획

1. 기존 테스트로 모델·표본 추출·평가 동작을 고정한다.
2. 공통 데이터는 `data/`에 유지하고 원본 실습자료를 `references/`로 옮긴다.
3. RetailHero·현재 Criteo·이전 Criteo의 노트북과 결과를 실험별로 모은다.
4. 재사용 코드를 `shared/`로 옮기고 노트북·생성기·실행기·테스트의 import와 경로를 수정한다.
5. 결과·데이터 해시, 노트북 생성과 경로 탐색, 회귀 테스트를 검증한다.

계산식이나 모델 조건의 리팩터링은 범위에 포함하지 않는다. 새로운 의존성을 추가하지 않는다.
새 호환 래퍼를 만들지 않고 호출부를 실제 위치로 수정한다.
표본 캐시 불일치 시 재생성하는 기존 동작은 코드/입력 식별에 필요한 동작이므로 유지한다.
작성과 별도로 읽기 전용 검토를 수행한다.

## 이동 대응

| 이전 | 현재 |
|---|---|
| `01. ML for Causal Inference_실습자료.ipynb` | `references/ML_for_Causal_Inference.ipynb` |
| `01_RetailHero.ipynb` | `experiments/01_retailhero/study.ipynb` |
| `02_Criteo.ipynb` | `experiments/02_criteo/study.ipynb` |
| `artifacts/causal_notebooks/retailhero/` | `experiments/01_retailhero/results/` |
| `artifacts/causal_notebooks/criteo/` | `experiments/02_criteo/archive/baseline_results/` |
| `artifacts/criteo_study/previous_02_Criteo.ipynb` | `experiments/02_criteo/archive/baseline.ipynb` |
| `artifacts/criteo_study/` | `experiments/02_criteo/results/` |
| `causal_data.py`, `causal_utils.py` | `shared/` 아래 같은 이름 |
| `criteo_study.py`, `criteo_models.py`, `criteo_inference.py` | `shared/criteo/study.py`, `models.py`, `inference.py` |
| `task.md` | `docs/research_plan.md` |

기존 Git 작업 트리에 이미 삭제되어 있던 과거 prototype·그래프 실험 파일은 복원하지 않는다.
기존 결과 파일 안의 소스 해시와 절대/상대 경로는 당시 기록으로 보존한다.
폴더 정리 자체는 모델 재실행이나 기존 가설 재평가를 포함하지 않는다.
