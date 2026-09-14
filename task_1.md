# Causal Discovery 최소 개발 명세

## 1. 목표

보유 중인 구리 시계열 데이터를 이용해 **시장 변수 간 시간차 인과관계를 탐색하고, 구리 가격으로 이어지는 주요 경로를 설명할 수 있는 최소 분석 프로토타입**을 만든다.

Kausalyze의 공개 구조를 참고해 다음 흐름을 우선 구현한다.

**Historical Time Series → Data Quality / Availability 처리 → Time-aware Causal Discovery → Domain Constraint → Causal Path Review → Result Interpretation**

초기 단계에서는 웹 서비스나 운영 시스템을 만들지 않는다. 모든 분석 과정과 결과를 하나의 `.ipynb`에서 확인할 수 있게 한다.

사용 데이터는 아래 파일을 기준으로 한다.

```text
/Users/hyeonee/workspace/causal_study/data/com_copper_raw.csv
```

---

## 2. 최소 개발 범위

| 영역              | 최소 구현 내용                                                           |
| --------------- | ------------------------------------------------------------------ |
| 데이터             | `/Users/hyeonee/workspace/causal_study/data/com_copper_raw.csv` 사용 |
| 변수              | 구리 가격 + 3개 이상 9개 미만의 핵심 설명변수로 시작                                           |
| 전처리             | 결측, 이상값, 파생변수 중복, 정상성, 시차, 데이터 입수지연 처리                             |
| Discovery       | 시계열 Causal Discovery 1개 이상 구현                                      |
| 비교              | 필요하면 2~3개 방법론 결과 비교                                                 |
| Prior Knowledge | 경제·시장 구조상 불가능하거나 명확한 관계를 제약조건으로 반영                                 |
| 결과              | 구리 가격으로 들어오는 인과경로, lag, 방향, 강도/신뢰도 확인                              |
| 검증              | 기간 변경 또는 bootstrap 등을 통한 관계 안정성 확인                                 |
| 출력              | 표·그래프·간단한 해석을 Notebook 셀에서 확인                                      |

처음부터 전체 변수를 사용하지 않는다. 최소 변수셋에서 결과가 해석 가능하고 반복적으로 나타나는지 확인한 뒤 변수를 단계적으로 추가한다.

---

## 3. Causal Discovery 방법

데이터 특성과 목적에 맞춰 방법을 선택한다.

우선 검토 대상은 다음과 같다.

* **PCMCI+, LPCMCI 등 제약 기반 계열**: lagged time-series causal discovery의 기본 후보, latent confounder 가능성이 중요한 경우 비교
* **VAR-LiNGAM / Granger 계열**: 단순한 선형 시계열 기준 결과 확인
* **DYNOTEARS 등 Score 기반의 Dynamic DAG 계열**: 구조학습 방식과 결과 비교가 필요할 경우 추가
* 그 외 논문이나 공개된 산업 적용 사례에서 현재 데이터에 적합한 방법이 확인되면 사용 가능
=> 구현 과정에서 논문이나 오픈소스 등 꼼꼼하게 읽은 후 구현 진행

초기에는 모든 알고리즘을 구현할 필요가 없다. **가장 단순한 방법 1개로 전체 흐름을 완성하고, 결과가 불안정하거나 방법론 가정이 문제가 될 때 다른 방법을 추가한다.**

방법론 비교 시 간선 개수보다 구리 가격으로 들어오는 경로의 안정성, 방향성, lag 일관성, 해석 가능성을 중요하게 본다.

---

## 4. Circularity 및 정보시점 처리

시장 데이터에는 `가격 → 재고`, `재고 → 가격`처럼 피드백이 존재할 수 있고, 실제 데이터 발표 시점 때문에 분석 당시 사용할 수 없었던 정보가 포함될 수 있다.

따라서 각 변수에 대해 다음 두 시점을 구분한다.

**observation period / reference period**
해당 값이 설명하는 경제적 시점

**availability time**
실제로 분석자가 해당 값을 사용할 수 있었던 시점

Causal Discovery용 데이터에서는 동일시점 정보누수, 미래 정보 사용, 가격과 사실상 동일한 파생변수, 회계적으로 연결된 중복변수를 점검한다.

가능하면 `raw timeline`과 `availability-adjusted timeline` 결과를 비교해 **현업에서 실제로 확인할 수 있었던 선행 관계가 무엇인지** 함께 확인한다.

인과관계 탐색과 가격 예측은 별개의 문제로 취급한다. Causal Discovery 결과를 예측 성능만으로 평가하지 않으며, 향후 필요할 경우 인과 변수의 실무적 정보가치를 별도의 forecasting 실험으로 검증한다.

---

## 5. Domain Knowledge 적용

Kausalyze가 공정 구조와 엔지니어 지식을 모델 제약에 사용하는 것처럼, 원자재 시장에서도 사전 지식을 활용한다.

명백히 불가능한 방향(설명이 어렵고 말도 안되는 관계), 데이터 정의상 결정되는 관계, 정보 발생 순서를 위반하는 관계는 차단할 수 있다.

경제적으로 해석 가능한 관계는 prior 또는 검토 대상으로 사용할 수 있으며, 확실한 근거가 있을 때만 간선을 고정한다.

초기 프로토타입에서는 LLM이 변수 정의와 시장 메커니즘을 바탕으로 다음을 작성한다.

`allowed edge / blocked edge / fixed edge / expected sign / rationale`

고정된 관계와 데이터에서 발견된 관계는 결과에서 구분해서 표시한다.

---

## 6. Notebook 최종 산출물

최소 Notebook은 아래 흐름까지만 완성한다.

`Load Data → Minimal Variable Selection → Availability Alignment → Preprocessing → Causal Discovery → Domain Constraint → Causal Graph → Target Upstream Paths → Stability Check → Interpretation`

최종적으로 다음 질문에 답할 수 있으면 1차 개발 완료로 본다.

**“현재 최소 변수셋에서 구리 가격과 연결되는 시간차 관계가 존재하는가, 해당 관계는 데이터 입수지연을 고려해도 유지되는가, 기간을 바꿔도 반복되는가, 경제적으로 설명 가능한가?”**

이 단계에서 관계가 충분히 안정적이지 않아도 결과 자체를 보존한다. 이후 변수 확대, regime 분석, causal effect estimation, counterfactual/what-if, forecasting, UI 연결은 별도 단계에서 추가한다.

/Users/hyeonee/workspace/causal_study/02. prototype.ipynb -> 여기 구현, 만약 추가 파일 설정 필요하면 추가 가능
