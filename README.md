# Bond Research Engine

운용사·학술 연구를 실행 가능한 Rule로 변환하고, 미국 채권 ETF로 백테스트한 뒤 PIMCO Active Bond ETF(BOND)와 비교하는 Streamlit 연구 플랫폼입니다.

```text
운용사·학술 연구 → Rule Library → ETF Mapping → Python Engine → Backtest → PIMCO BOND 역설계
```

기본 운용 유니버스는 SHY, IEF, TLT, TIP, LQD, HYG, MBB의 7개 ETF입니다. Structured Proxy 실험군은 BOND의 구조화채권 빈칸을 보기 위해 JAAA, BKLN, CMBS를 별도로 추가합니다.

## 실행

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

브라우저에서 `http://localhost:8501`을 엽니다. 데이터는 15분 동안 메모리 캐시되며, 사이드바의 **데이터 새로고침** 버튼으로 즉시 갱신할 수 있습니다.

## FRED API 키

FRED API 키는 코드에 넣지 않습니다. `.streamlit/secrets.toml.example`을
`.streamlit/secrets.toml`로 복사한 뒤 본인의 키를 입력하세요.

```toml
FRED_API_KEY = "your_fred_api_key"
```

환경변수 `FRED_API_KEY`를 사용해도 됩니다. 로컬 비밀파일은 `.gitignore`에 포함되어 배포물이나 Git에 들어가지 않습니다.

사이드바의 **FRED 실시간 갱신 시도**는 기본적으로 꺼져 있습니다. 끈 상태에서는 번들 스냅샷으로 즉시 실행되고, 켜면 API 키를 사용해 최신값을 요청한 뒤 실패 시 스냅샷으로 복구합니다.

## Streamlit Community Cloud 배포

1. 이 폴더를 비공개 GitHub 저장소에 올립니다.
2. Streamlit Community Cloud에서 저장소와 `app.py`를 선택합니다.
3. 앱의 **Settings → Secrets**에 아래 값을 등록합니다.

```toml
FRED_API_KEY = "your_fred_api_key"
```

배포된 앱에는 `https://...streamlit.app` 형태의 주소가 생성됩니다. 로컬 `secrets.toml`은 저장소에 올리지 않습니다.

## 데이터 의미

- ETF: Yahoo Finance 공개 chart endpoint의 조정가격 및 지연시세
- 금리·스프레드·거시지표: FRED 공식 API
- 복제 목표: PIMCO Active Bond ETF(BOND) 공개 총수익 가격
- 시장 벤치마크: iShares Core U.S. Aggregate Bond ETF(AGG)
- 인터넷 장애 시 `.cache/`의 마지막 정상 데이터 사용
- 새로 압축을 푼 폴더처럼 캐시가 없고 API 키도 없으면 `data/fred_snapshot.csv`로 즉시 시작합니다.
- ETF 요청 실패 시 `data/etf_total_return_snapshot.csv`로 자동 대체합니다.
- 번들 스냅샷 기준일은 화면의 Source Status에서 확인하며 실시간 값으로 해석하지 않습니다.

## 백테스트 해석

- 모든 필수 시계열과 ETF가 실제로 겹치는 날짜부터 자동 시작합니다.
- 월말 신호를 익월 총수익에 적용하고 설정한 거래비용을 차감합니다.
- 메인 성과표의 Correlation, Beta, Tracking Error, Information Ratio는 모두 AGG 대비 기준입니다. BOND는 성과 벤치마크가 아니라 복제 목표와 포지션 역설계 기준으로 별도 표시합니다.
- Alpha Replica v2는 새 ETF를 추가하지 않고 7개 ETF 안에서 MBS RV, Credit Cushion, Duration/MOVE, Curve Roll-down, Real Yield 점수를 동시에 비중에 반영합니다.
- Structured Proxy Engine은 Alpha v2에 JAAA(AAA CLO), BKLN(senior loan), CMBS(commercial MBS)를 최대 20%까지 얹어 Non-Agency MBS/ABS/CLO 빈칸을 ETF로 근사하는 실험군입니다.
- Structured Proxy v2 Candidate는 structured sleeve cap, JAAA/BKLN/CMBS 구성, Light/Medium/Strong risk gate 후보를 비교한 뒤 선택한 인샘플 후보입니다. 현재 표본에서는 `20% cap · Carry Tilt · Strong Gate`가 선택됩니다.
- MBS RV는 `30년 모기지 금리 - 10년 국채금리`의 5년 백분위, MBB-IEF 3개월 상대수익, MOVE 방향성을 함께 봅니다.
- Duration/MOVE는 10년 금리 3개월 변화, TLT 6개월 모멘텀, 디스인플레이션/물가상승 레짐, MOVE 극단값을 함께 사용합니다.
- Credit Cushion은 HY OAS·IG OAS의 5년 백분위, HY 스프레드 3개월 변화, HYG/LQD 상대가격 확인을 함께 사용합니다. 실제 신규발행 할인이나 CLO 종목선택을 직접 관찰한 값은 아닙니다.
- Curve Roll-down은 5-10년/10-30년 커브와 10Y-2Y steepening 여부를 사용하고, Real Yield는 10년 실질금리 백분위와 TIP-IEF 상대수익을 사용합니다.
- Alpha Source Attribution은 전월 말 점수판이 다음 달 Alpha Return, AGG 대비 초과수익, BOND 대비 격차를 얼마나 설명했는지 Dominant Alpha별·점수별로 보여줍니다.
- 일반 알파 목표는 분기말에만 변경합니다. HY 스프레드 충격, `VIX 3년 백분위 90% 이상 또는 MOVE ≥ 110 및 3년 백분위 85% 이상`, 실업 악화, NFCI 스트레스, HYG 가격붕괴 중 3개 이상이면 일별로 SHY 100% 전환하고 최소 20거래일과 5일 연속 해제 후 재진입합니다.
- 대시보드는 같은 v2 규칙에서 `VIX 또는 MOVE`, VIX만, MOVE만 사용한 반사실 성과를 함께 표시합니다.
- 위 임계값은 35개월 표본에서 확정된 규칙이 아니라 향후 표본 외 검증이 필요한 가설입니다.
- BOND Exposure Replica는 `SHY 5% / IEF 25% / TLT 5% / TIP 5% / LQD 15% / HYG 5% / MBB 40%`를 전략적 출발점으로 사용합니다.
- Replica는 듀레이션 5.5~6.5년, MBB 30~45%, SHY 최대 15%를 유지하며 분기별 일반 조정은 ETF당 최대 5%p입니다.
- Risk-Off에는 현금으로 전면 회피하지 않고 LQD·HYG를 줄여 IEF·MBB로 이동합니다. 이는 BOND의 코어 위험 구조를 유지하기 위한 근사 규칙입니다.
- Hybrid Engine은 Carry·Value·Real Yield·Curve·Credit·Regime을 월별, Momentum을 주별, Risk-Off를 일별로 관찰합니다.
- 일반 신호는 5거래일 지속, 최소 20거래일 보유, 목표비중 차이 10%p 이상일 때만 거래하며 진입 점수(+0.15)와 청산 점수(-0.05)를 다르게 적용합니다.
- 한 번의 리밸런싱에서 ETF별 변화는 최대 15%p, 최근 1년 편도 회전율은 최대 100%이며 Risk-Off 전환은 이 제한을 건너뜁니다.
- Sharpe는 3개월 T-Bill을 차감한 초과수익 기준이므로 명목수익이 양수여도 음수가 될 수 있습니다.
- FRED revised data를 사용하므로 최종 연구에서는 ALFRED vintage 또는 최초 발표 데이터로 재검증해야 합니다.
- BOND return-implied exposure는 실제 보유비중이 아니라 24개월 rolling ridge regression 근사치입니다.
- 실제 포지션 역설계는 SEC Form N-PORT의 BOND 시리즈(`S000033233`) 분기말 전체 보유내역을 사용합니다.
- Effective Duration은 N-PORT 통화별 DV01 합계 ÷ 순자산 × 10,000으로 계산합니다. 2026-03-31 값 6.53년은 PIMCO 공식 팩트시트와 일치합니다.
- Sector Mix는 파생상품 명목노출을 제외한 funded holdings 기준 프록시이므로 PIMCO의 Gross Market Value sector allocation과 정의가 다릅니다.
- N-PORT에는 신용등급 필드가 없어 Government+Agency MBS를 High Quality Proxy, Non-Agency MBS+ABS/CLO+Credit/Loans를 Spread Risk Proxy로 표시합니다. 실제 평균 신용등급이 아닙니다.
- MBS RV Gap Diagnostic은 BOND의 Agency MBS와 우리 MBB 비중 차이, 그리고 ETF 유니버스로 직접 복제하지 못하는 Non-Agency MBS+ABS/CLO 구조화채권 빈칸을 분리해 보여줍니다.
- Structured Proxy를 반영하면 이 빈칸이 얼마나 줄어드는지 `Rule Structured Proxy`와 `Total Gap After Structured Proxy`로 확인합니다. 다만 CLO/loan ETF는 신용·유동성 위험이 있어 확정 규칙이 아니라 실험 가설입니다.
- v2 후보는 빠른 월말 grid로 후보를 고르고, 선택된 조합은 다시 일별 위험 게이트 백테스트로 계산합니다. 인샘플 선택 편향이 있으므로 다음 공시·다음 시장국면에서 고정 규칙으로 검증해야 합니다.
- 공개되지 않은 과거 월은 보간하지 않습니다. SEC 역사는 분기말 기준이며, PIMCO 일별 holdings는 앞으로 월말 스냅샷을 누적해야 월별 역사가 됩니다.

## Observed IF-THEN Rulebook

- 2019년 9월 이후 BOND N-PORT의 다음 분기 포지션 변화를 예측 대상으로 사용합니다.
- 시장 입력은 국채금리, 실질금리, 커브, IG/HY OAS, 모기지 스프레드, 물가·성장·실업, VIX, NFCI입니다.
- Duration, 5Y/10Y/30Y Key Rate Duration, Agency MBS, Credit, Gross/Net, 금리파생 명목금액의 변화를 각각 탐색합니다.
- 시간순 앞 70%에서 시장 변수의 25·33·50·67·75 분위수를 탐색하고, 뒤 30%는 임계값을 다시 맞추지 않는 OOS 구간으로 둡니다.
- 학습 구간에서 최소 4개 조건 관측, 적중률 55%, baseline 대비 Lift 8%p를 요구합니다.
- 각 Rule에는 학습 표본 수·적중률·baseline·Lift, OOS 표본 수·적중률, binomial p-value, 탐색횟수 보정값, 경제적 논리를 함께 표시합니다.
- 후보 A/B는 상대적 증거등급이지 내부 규칙의 확인을 뜻하지 않습니다. 새 공시가 나올 때 임계값을 고정한 채 순차 검증해야 합니다.

이 앱은 교육 및 리서치 도구이며 투자자문이나 자동주문 시스템이 아닙니다.

## 운영 메모

- 기본 실행은 실데이터 모드입니다.
- ETF는 장중 실시간 체결가가 아니라 공개 엔드포인트의 지연시세일 수 있습니다.
- FRED 데이터는 각 지표의 발표 주기에 따라 갱신됩니다.
- 자동 운용에 사용하기 전 데이터 라이선스, point-in-time 데이터, 거래비용을 별도로 검토해야 합니다.
