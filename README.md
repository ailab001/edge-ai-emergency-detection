# Edge AI Emergency Detection Platform

Edge AI 기반 응급 상황 탐지 플랫폼

## Overview

센서 데이터 기반 이상징후 탐지를 위한 경량 Edge AI 아키텍처

Originally developed for home-based emergency safety service.

**주요 특징:**
- 게이트웨이 레벨 실시간 이상 감지 (Edge AI)
- 다중 센서 융합 기반 복합 판단
- 경량 예측 모델 (Python 프로토타입)
- Docker 기반 개발 환경

## Architecture

```
[센서 4종] → [게이트웨이(Edge AI)] → [수집서버] → [Core AI] → [관리시스템]
```

### Edge AI Components

1. **Expert System**: 화재·응급호출 등 명확한 위험 상황 즉각 판단
2. **Pattern Matching**: 비지도 학습 기반 이상 패턴 감지
3. **Prediction Model**: 경량 예측 모델 (GPS, Trend, Anomaly)

## Project Structure

```
edge-ai-emergency-detection/
├── edge_ai/
│   ├── expert_system/      # 규칙 기반 이벤트 감지
│   ├── pattern_matching/   # 비지도 학습 패턴 매칭
│   ├── prediction/         # 경량 예측 모델
│   ├── datasets/           # 데이터 생성 및 시뮬레이션
│   └── tests/              # 단위 테스트 및 통합 테스트
├── shared/                 # 공유 리소스
├── docs/                   # 프로젝트 문서
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

## Requirements

- Python >= 3.8
- Docker & Docker Compose
- scikit-learn >= 1.3
- pydantic >= 2.0
- numpy >= 1.24

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd edge-ai-emergency-detection
```

### 2. Start the Docker environment

```bash
make up
```

## Usage

### Run Tests

```bash
make test
```

### Generate Sample Data

```bash
make generate
```

### Run Scenarios

```bash
make run-scenarios
```

### Run Simulation

```bash
# Run specific scenario
make simulate SCENARIO=fire_detection

# Run specific cycle
make simulate CYCLE=10
```

### Interactive Shell

```bash
make shell
```

## Development

### Running Individual Modules

```bash
# Expert System
docker compose exec edge python -m edge_ai.expert_system.engine

# Pattern Matching
docker compose exec edge python -m edge_ai.pattern_matching.pipeline

# Prediction
docker compose exec edge python -m edge_ai.prediction.pipeline
```

### Running Tests

```bash
# All tests
docker compose exec edge python -m pytest edge_ai/tests/ -v

# Specific test file
docker compose exec edge python -m pytest edge_ai/tests/test_expert_system.py -v
```

## Documentation

자세한 구현 지침 및 아키텍처 설명은 `docs/` 디렉토리를 참조하세요:

- [PROJECT_GUIDELINES.md](docs/PROJECT_GUIDELINES.md): 전체 프로젝트 구현 지침
- [WORK_ORDER_GUIDE.md](docs/WORK_ORDER_GUIDE.md): 작업 가이드

## Clean Up

```bash
make clean
```

## Disclaimer

This is a prototype developed for educational and technical demonstration purposes.
Not intended for production use without proper validation and testing.

본 프로젝트는 기술 시연 및 교육 목적의 프로토타입입니다.
프로덕션 환경에서 사용 시 적절한 검증과 테스트가 필요합니다.

## License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
