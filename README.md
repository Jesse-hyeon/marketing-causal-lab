# 마케팅 데이터 인과추론 실험실

마케팅 데이터에 인과추론과 Causal ML을 적용하고, 가설·효과 추정·고객 타기팅 정책을 비교합니다.
`ML for Causal Inference` 실습자료를 참고하며 S/T/X-Learner, DML, DR 등의 방법을 노트북에서 실습합니다.

## 폴더 안내

```text
causal_study/
├── data/                              # 모든 실험에서 공유하는 원본·출처 기록
│   ├── retailhero/
│   └── criteo/
├── references/
│   └── ML_for_Causal_Inference.ipynb   # 참고용 원본 실습자료
├── experiments/
│   ├── 01_retailhero/
│   │   ├── study.ipynb
│   │   └── results/
│   ├── 02_criteo/
│   │   ├── study.ipynb                 # 현재 H3/H4 실험
│   │   ├── results/
│   │   └── archive/                   # 이전 H1/H2 실험·결과
│   └── 03_meta_learner_comparison/     # 향후 방법별 비교 실험
├── shared/
│   ├── causal_data.py                 # 공통 데이터 로딩·표본 추출
│   ├── causal_utils.py                # S/T/X, DML·DR, 평가 함수
│   └── criteo/                        # Criteo 전처리·모델·평가 코드
├── scripts/                           # 노트북 생성·일괄 실행
├── tests/
├── docs/
└── requirements.txt
```

## 시작하기

프로젝트 루트에서 Python 환경을 준비합니다.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name causal-study --display-name 'Python (causal_study .venv)'
```

원본 데이터는 용량 때문에 Git에 포함하지 않습니다. 파일 구성·고정 버전 URL·검증값은
[data/retailhero](data/retailhero/README.md)와 [data/criteo](data/criteo/README.md)에 있습니다.
각 데이터 manifest의 URL에서 내려받아 해당 폴더에 보관하세요.

노트북에서 `causal-study` 커널을 선택합니다. 루트 또는 프로젝트 안의 노트북 폴더에서
실행하면 상위 디렉터리를 탐색해 공통 코드·데이터를 찾습니다.

```sh
# 원하는 실험 하나 실행 (실제 모델 학습과 결과 저장)
.venv/bin/python scripts/execute_notebooks.py experiments/01_retailhero/study.ipynb
.venv/bin/python scripts/execute_notebooks.py experiments/02_criteo/study.ipynb

# 작은 합성/fixture 데이터 기반 테스트
.venv/bin/python -m pytest -q
```

`execute_notebooks.py`에 인수가 없으면 RetailHero와 현재 Criteo를 순서대로 실행합니다.
각 실험은 `data/`를 공유하고 설정·표·결과는 자신의 `results/`에 저장합니다.
아직 시작하지 않은 방법 비교 실험의 범위는 해당 폴더 README에 기록했습니다.

## 노트북 재생성

```sh
.venv/bin/python scripts/build_notebooks.py
.venv/bin/python scripts/build_criteo_study.py
```

생성기는 노트북 파일을 덮어쓰며 기존 셀 실행 결과를 없앱니다. 코드 재생성이 필요할 때만 사용하세요.
첫 명령은 RetailHero와 현재 Criteo 노트북을, 두 번째는 현재 Criteo만 생성합니다.
이전 Criteo baseline은 `build_notebooks.build('criteo')`를 명시적으로 호출할 때만 재생성합니다.

## 결과 해석·보존

- 원본 실습자료와 기존 수치 결과는 폴더 이동 시 그대로 보존했습니다.
- `results/code_versions.json`, 기존 검증 JSON의 경로·해시는 당시 실행을 기록한 값입니다.
  현재 파일의 해시와 같다는 의미는 아니며, 새 실행 시 새 기록이 생성됩니다.
- 기존 표본 캐시도 로컬에 보존했습니다. 코드 변경으로 키가 달라지면 새 캐시를 생성합니다.
- 데이터·캐시·가상환경·로컬 OMX 상태는 푸시에서 제외합니다.
- 기존 연구 명세는 [docs/research_plan.md](docs/research_plan.md), 정리 기록은
  [docs/reorganization.md](docs/reorganization.md)에 있습니다.
