# Criteo Uplift v2.1

출처: [Hugging Face 공식 저장소](https://huggingface.co/datasets/criteo/criteo-uplift). 원본 데이터 카드의 사본은 `SOURCE_README.md`에 보관했다.

## 파일과 검증

- 파일: `criteo-research-uplift-v2.1.csv.gz`
- 다운로드일: 2026-11-10
- 원본 revision: `2424920019e49d52d72c13ac1143ec5d53af276b`
- 압축 크기: **311,422,618 bytes**, 약 **311 MB / 297 MiB**
- 압축 해제 시 내용 크기: **3,248,115,221 bytes**, 약 **3.25 GB**. 해제본은 저장하지 않았다.
- 실제 파일 확인: **13,979,592개 데이터 행, 16개 열**. 전체 gzip 스트림을 읽어 CRC를 검증했고, SHA-256을 Hugging Face 메타데이터와 대조했다. pandas로 첫 100,000행 읽기도 확인했다.
- 고정 버전 URL과 SHA-256은 `criteo_manifest.json`에 기록했다.
- 원본 카드 소개에는 25M 행 / 11개 피처라는 설명도 있으나, 이 v2.1 파일은 실제 확인한 13,979,592행과 12개 피처를 기준으로 사용한다.
- 원본 카드의 라이선스 표기: `CC-BY-NC-SA-4.0`.

## 열과 인과추론에서의 의미

| 열 | 의미 |
|---|---|
| `f0`–`f11` | 익명화·변환된 수치 피처 12개 |
| `treatment` | 광고 타기팅 처치군 배정 여부 |
| `conversion` | 전환 여부 |
| `visit` | 방문 여부 |
| `exposure` | 실제 광고 노출 여부 |

원본 설명상 여러 무작위 incrementality 실험을 합친 데이터다. `treatment`와 `exposure`는 서로 다른 변수다. 배정 효과를 분석할 때는 `treatment`를 처치로 사용하고 `exposure`를 사전 고객 피처로 넣지 않는다.

원본은 개인정보 보호를 위해 비균등 서브샘플링되어 실제 원 실험의 절대 증분 효과를 복원할 수 없다고 설명한다. 따라서 이 자료에서 산출한 효과 크기를 실제 광고 캠페인의 ROI로 그대로 해석하지 않는다. 광고비나 전환당 기여이익은 제공 열에 없다.

## 메모리를 아끼며 읽기

프로젝트 루트에서 기존 `.venv`의 pandas로 실행할 예시:

```python
from pathlib import Path
import pandas as pd

path = Path("data/criteo/criteo-research-uplift-v2.1.csv.gz")
sample = pd.read_csv(path, nrows=100_000)
```

위 코드는 파일 앞부분 확인용이며 무작위 표본은 아니다. 분석 표본이 필요하면 파일 전체에서 별도로 무작위 추출한다.

전체를 처리할 때는 압축 상태로 청크 단위로 읽는다. 필요한 열만 선택하고, 처리한 청크를 모두 메모리에 모으지 않는다.

```python
feature_columns = [f"f{i}" for i in range(12)]
columns = feature_columns + ["treatment", "conversion", "visit"]
dtypes = {column: "float32" for column in feature_columns}
dtypes.update({column: "int8" for column in ["treatment", "conversion", "visit"]})

with pd.read_csv(path, usecols=columns, dtype=dtypes, chunksize=100_000) as chunks:
    for chunk in chunks:
        # 필요한 집계·표본 추출을 수행하고 다음 청크로 진행한다.
        pass
```

이번 작업은 원본 확보와 읽기·무결성 검증까지이며 모델 학습이나 정책 효과 검증은 아직 수행하지 않았다.
