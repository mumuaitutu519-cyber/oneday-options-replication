# 하루 만기 비트코인·이더 옵션 재현 패키지

기준 원고: `OneDay_LEEKangRyu2026.11.06_tables_figures.docx`, 표 1–10, 그림 1–3. 원고는 수정하지 않았습니다. 유료 데이터 공급자 자료는 확인되지 않았습니다.

Python 3.12와 Bash가 있는 노트북에서 압축을 풀고 다음 한 줄을 실행하십시오.

```bash
bash run_all.sh --offline
```

가상환경과 고정 버전 의존성을 자동 설치하고 모든 표·그림과 원고 대조표를 생성합니다. 기본 **빠른 모드**는 보관된 일별 예측·VaR 입력을 사용하여 손실, 검정, 회귀, MCS, 부트스트랩 구간과 그림을 다시 계산합니다. **전체 모드**는 다음 명령으로 120개 사양의 롤링 모형과 VaR 잔차 분위수까지 다시 추정합니다.

```bash
bash run_all.sh --full --offline
```

폴더: `code/` 최종 계산 코드, `data/` 압축 분석 자료, `raw_rebuild/` 원자료 추출 코드·출처, `output/` 재계산 표·그림, `verify/` 원고 셀 대조 및 차이, `logs/` 실행 기록, `wheels/` 오프라인 설치 파일. 변수 정의는 `DATA_DICTIONARY.csv`에 있습니다.

|표·그림|패키지 스크립트|자료|
|---|---|---|
|표 1|`tables.py: tables_premia`|`premium_*`|
|표 2–3|`estimate.py`, `tables.py: tables_forecasts`|`features_*`, `forecast_inputs/`, `forecast_cache/`|
|표 4|위와 같음|`*matrix*` 예측 입력·경로|
|표 5|`tables.py: tables_premia`|`premium_*`|
|표 6–7|`tables.py: decomposition`|`decomposition_*`, `numerical_grid_diagnostics_*`|
|표 8|`tables.py: tables_trading`|`straddle_daily_*`, `strategy_availability_*`|
|표 9|`tables.py: tables_trading`, `tables_premia`|`atm_trade_daily_by_side`, `trade_quality_*`, `premium_*`, `straddle_daily_*`|
|표 10|`tables.py: tables_risk`, `estimate.py`|`risk_inputs_*`, 예측 입력·경로|
|그림 1–3|`figures.py`|`premium_*`|

스크립트는 `code/`, 자료는 `data/` 아래에 있습니다. 원 프로젝트의 **표·그림 ↔ 실제 버전 스크립트 ↔ 결과 파일** 상세 대응표는 `verify/TABLE_FIGURE_LINEAGE.csv`입니다. ROUND10은 다른 논문의 분석이며 이 원고의 입력이 아닙니다.

실측 시간: **새 가상환경 오프라인 설치 포함 61초, 재실행 41초**, 전체 **120개 사양 신규 재추정 627초(10분 27초), 최종 정의 대조 재실행 38초**. 4개 프로세스, 수치 라이브러리는 프로세스당 1개 스레드를 사용합니다. 빠른 모드는 30분보다 훨씬 짧습니다. 제공 wheel은 **macOS Apple Silicon / Python 3.12**용입니다. 다른 플랫폼은 Python 3.12와 Bash에서 `bash run_all.sh`로 해당 플랫폼의 고정 버전 패키지를 인터넷으로 설치할 수 있으나, 그 플랫폼의 오프라인 실행은 검증하지 않았습니다.

`output/TABLE_1.md`–`TABLE_10.md`와 CSV, `output/FIGURE_COMPARISON.html`, `verify/*_side_by_side.png`를 확인하십시오. 그림의 CSV 원자료도 `output/`에 있습니다. 모든 숫자는 원고의 표시 소수 자리에서 반올림하여 비교합니다. 숫자 분모와 MCS 문자 검증은 별도로 기록합니다.

기본 출력은 원고에 명시된 하루 만기 23시간과 2020년 1월 이후 표본을 적용합니다. 이에 따라 표 1·5·9의 숫자 202개와 유의성 별표 7곳이 달라집니다. 원 결과의 하루 만기 자료는 실제로 24시간 보간치를 사용했고, 표 9A에는 2020년 이전 거래가 포함되었습니다. 원 버전 재계산은 `output/source_version/`, `verify/source_version/`에 별도로 보존했으며, 원고 숫자 1,460개와 MCS 문자 셀 52개가 모두 일치합니다. 변경 셀과 값은 `verify/MANUSCRIPT_CORRECTIONS.csv`, 원인은 `verify/MISMATCH_REPORT.md`에 있습니다. 명시 정의를 적용한 하루 만기 mark 계산은 거래소의 underlying forward를 사용하며, 원 R5는 index를 forward 대용치로 사용했습니다. 더 긴 만기와 명시적으로 24시간인 분해표는 해당 표 주석을 따릅니다. 전체 모드도 일별 분산·손익 자료부터 시작하며, 대용량 분·거래 자료의 추출은 선택 사항입니다.

SHA-256은 `CHECKSUMS.sha256`, 파일 용도와 대응 표·그림은 `MANIFEST.csv`에 있습니다. 원자료의 이용 조건은 각 거래소에 따릅니다.

## GitHub 배포

원고는 PDF만 허용하는 배포 지침에 따라 DOCX를 제외했습니다. 기존 표·본문·그림 추출 자료는 그대로 보존하고 해시를 확인합니다. 계산식, 분석 자료, 결과 및 202개 수치 차이 검증 CSV는 변경하지 않았습니다. 자세한 내용은 `PACKAGING_NOTES.md`를 참조하세요.
