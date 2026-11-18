# RetailHero Uplift 데이터

출처: [Hugging Face](https://huggingface.co/datasets/pytorch-lifestream/retailhero-uplift/tree/main/data) · [공식 대회](https://ods.ai/competitions/x5-retailhero-uplift-modeling) · [공식 데이터 설명](https://ods.ai/competitions/x5-retailhero-uplift-modeling/data)

## 저장 및 검증

- 2026-11-10 다운로드. 원본 revision: `7744156a4f89f4828607921e8c9668f04801f4de`.
- 압축 원본 6개, 총 **622,956,732 bytes = 623 MB = 594 MiB**.
- 모든 파일의 바이트 수와 SHA-256을 Hugging Face 저장소 메타데이터와 대조했다.
- 파일별 고정 버전 URL, 크기, SHA-256은 `retailhero_manifest.json`에 기록했다.
- 압축을 해제하거나 별도 Hugging Face 캐시를 만들지 않았다. 원본 `.csv.gz`는 Git에서 제외한다.

| 파일 | 압축 크기(MB, 10진수) | 내용 |
|---|---:|---|
| `clients.csv.gz` | 7.63 | 고객 400,162명: 고객 ID, 카드 발급·첫 사용 시각, 나이, 성별 |
| `uplift_train.csv.gz` | 1.18 | 학습 고객 200,039명: `client_id`, `treatment_flg`, `target` |
| `uplift_test.csv.gz` | 1.07 | 테스트 고객 200,123명의 ID; 처치·성과 레이블 없음 |
| `purchases.csv.gz` | 608.83 | SMS 캠페인 이전 구매 이력; 상품별 행 포함 |
| `products.csv.gz` | 1.05 | 상품 43,038개와 분류·속성 |
| `uplift_sample_submission.csv.gz` | 3.18 | 대회 제출 형식 예시 200,123행; 정답이나 학습용 성과가 아님 |

## 공간과 메모리를 아끼며 읽기

기존 `.venv`에 pandas가 설치되어 있다. 프로젝트 루트에서 실행할 예시:

```python
from pathlib import Path
import pandas as pd

data_dir = Path("data/retailhero")
clients = pd.read_csv(data_dir / "clients.csv.gz", dtype={"client_id": "string"})
train = pd.read_csv(data_dir / "uplift_train.csv.gz", dtype={"client_id": "string"})
df = train.merge(clients, on="client_id", how="left", validate="one_to_one")
```

두 원본은 합계 약 8.8MB다. 우선 이 조합으로 전환율과 고객별 uplift 분석을 시작할 수 있다. pandas는 `.gz`를 자동으로 읽으므로 사전 압축 해제가 필요 없다. 단, 압축 파일 크기와 읽은 뒤 RAM 사용량은 다르다.

구매 이력은 처음부터 전체를 DataFrame으로 읽기보다 청크 단위로 고객별 구매 빈도·최근 구매일 등의 피처를 집계한다.

```python
purchase_chunks = pd.read_csv(
    data_dir / "purchases.csv.gz",
    usecols=["client_id", "transaction_id", "transaction_datetime"],
    dtype={"client_id": "string", "transaction_id": "string"},
    chunksize=200_000,
)
for chunk in purchase_chunks:
    # 고객별 집계 결과를 누적한다. 전체 chunk를 리스트에 모으지 않는다.
    pass
```

이는 읽기 방식 예시이며 피처 집계 구현은 아직 포함하지 않는다. 거래 ID가 여러 상품 행에 반복될 수 있으므로 행 수를 거래 수로 해석하거나 `purchase_sum`을 행마다 단순 합산하기 전에 데이터 단위를 확인해야 한다.

반복 분석 시 필요한 고객별 집계만 작은 파일로 저장하는 편이 실용적이다. 원격 스트리밍도 가능하지만 압축 CSV를 반복 분석하면 재전송·재파싱 비용이 발생할 수 있다. 현재 크기에서는 압축 원본을 로컬에 한 번 저장하고 필요한 열·행 단위로 읽는 방식을 권장한다.

## 마케팅 의사결정에서의 의미

- 처치는 **광고 SMS 발송 여부**, 성과는 **이후 구매 여부**다. 검색·소셜 광고의 채널별 예산이나 입찰 데이터는 없다.
- 적합한 첫 질문은 “추가 구매를 유도할 가능성이 큰 고객에게만 유료 메시지를 보내면 어떤가?”다.
- `uplift_test`에는 성과가 없으므로 효과와 정책의 검증은 `uplift_train`을 고객 단위로 학습/검증 분리해서 진행한다.
- 원 대회는 uplift 순위 상위 30%의 추가 전환으로 평가한다. 이익 극대화에는 효과 크기 검증과 SMS 단가·할인비·구매당 기여이익 가정이 추가로 필요하다. 과거 구매금액은 캠페인 이후 추가 매출이 아니다.
- 공개 설명에서 무작위 배정 방식·확률은 확인하지 못했다. 효과를 인과적으로 해석하기 전에 처치 배정 근거와 필요한 가정을 점검한다.
- 고객 속성·날짜의 이상치, 결측치, 의사결정 시점 이후 정보 누출을 점검한 뒤 피처로 사용한다.

이번 작업 범위는 원본 확보·무결성 검증·읽기 확인까지이며, 인과 효과 추정이나 정책 성과 검증은 아직 수행하지 않았다.
