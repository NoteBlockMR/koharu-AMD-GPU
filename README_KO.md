<h1 align="center">Koharu AMD GPU</h1>

<p align="center"><b>Rust</b>로 작성된 머신러닝 기반 만화 번역기</p>

<p align="center"><a href="README.md">English</a> | 한국어</p>

> [!IMPORTANT]
> 이 저장소는 [mayocream/koharu](https://github.com/mayocream/koharu)를 기반으로
> AMD GPU용 ncnn Vulkan 가속을 실험적으로 추가한 포크입니다. 원본 Koharu 프로젝트와
> AMD 패치 포크는 별개이며, 패치 관련 문제는 이 포크의 이슈에 보고해 주세요.

Koharu는 만화 번역 과정을 자동화하는 로컬 우선 워크플로를 제공합니다. 객체 감지,
OCR, 인페인팅, LLM을 하나의 데스크톱 애플리케이션으로 결합하여 자연스러운 번역 환경을
구성합니다.

내부적으로 고성능 추론에는 [candle](https://github.com/huggingface/candle)과
[llama.cpp](https://github.com/ggml-org/llama.cpp)를 사용하며, 데스크톱 앱은
[Tauri](https://github.com/tauri-apps/tauri)로 제작되었습니다. 주요 구성 요소는 안전성과
성능을 위해 Rust로 작성되어 있습니다.

> [!NOTE]
> Koharu는 개인정보 보호를 위해 비전 모델과 로컬 LLM을 사용자 컴퓨터에서 실행합니다.

---

![Koharu 화면](docs/en-US/assets/koharu-screenshot-en.png)

지원과 커뮤니티 토론은 원본 프로젝트의
[Discord 서버](https://discord.gg/mHvHkxGnUY)에서 확인할 수 있습니다.

## AMD Vulkan 포크의 변경 사항

이 포크는 Windows의 AMD GPU에서 다음 엔진을 ncnn Vulkan으로 실행하도록 수정했습니다.

- 만화 텍스트 감지 및 분할
- 말풍선 영역 분할
- YuzuMarker 글꼴 및 색상 감지
- LaMa 만화 인페인팅

LaMa는 GPU와 시스템 메모리 사용량을 줄이기 위해 256×256 저메모리 모델을 사용하며,
LaMa를 로드하기 전에 다른 캐시 엔진을 해제합니다. AMD Radeon RX 9060 XT에서 시험했지만
다른 AMD GPU에서는 성능과 안정성이 다를 수 있습니다.

> [!WARNING]
> ncnn 모델과 `ncnn.dll`은 용량 및 재배포 문제로 Git 저장소에 포함되어 있지 않습니다.
> Vulkan 백엔드를 사용하려면 필요한 런타임 아티팩트를 `temp/vulkan-pilot`에 배치하거나
> `KOHARU_NCNN_ROOT` 환경변수로 해당 폴더를 지정해야 합니다. 파일이 없으면 일부 엔진이
> CPU로 폴백할 수 있습니다.

## 주요 기능

- 텍스트 영역, 말풍선 및 정리 마스크 자동 감지
- 만화 대사, 캡션 및 페이지 내 텍스트 OCR
- 원문 글자를 제거하는 인페인팅
- 로컬 또는 원격 LLM을 통한 번역
- 세로쓰기 CJK 및 RTL 언어를 지원하는 고급 텍스트 렌더링
- 편집 가능한 텍스트를 포함한 레이어 PSD 내보내기
- 자동화를 위한 로컬 HTTP API 및 MCP 서버

설치와 최초 실행 방법은 원본 문서의
[Koharu 설치](https://koharu.rs/how-to/install-koharu/) 및
[첫 페이지 번역](https://koharu.rs/tutorials/translate-your-first-page/)을 참고하세요.

## 사용법

### 단축키

캔버스:

- <kbd>Ctrl</kbd> + 마우스 휠: 확대/축소
- <kbd>Ctrl</kbd> + 드래그: 캔버스 이동

도구:

- <kbd>V</kbd>: 선택 도구
- <kbd>M</kbd>: 블록 도구
- <kbd>B</kbd>: 브러시 도구
- <kbd>E</kbd>: 지우개 도구
- <kbd>R</kbd>: 복구 브러시 도구
- <kbd>[</kbd> / <kbd>]</kbd>: 브러시 크기 감소/증가

기록 및 선택:

- <kbd>Ctrl</kbd>/<kbd>Cmd</kbd> + <kbd>Z</kbd>: 실행 취소
- <kbd>Ctrl</kbd>/<kbd>Cmd</kbd> + <kbd>Shift</kbd> + <kbd>Z</kbd>: 다시 실행
- <kbd>Ctrl</kbd>/<kbd>Cmd</kbd> + <kbd>A</kbd>: 현재 페이지의 모든 텍스트 블록 선택

전체 목록은 [키보드 단축키](https://koharu.rs/reference/keyboard-shortcuts/)를 참고하세요.

### 내보내기

현재 페이지를 하나의 렌더링 이미지 또는 레이어가 보존된 Photoshop PSD로 내보낼 수
있습니다. PSD에는 보조 레이어와 편집 가능한 번역 텍스트가 포함되어 후처리와 수동
보정에 유용합니다.

자세한 내용은
[페이지 내보내기 및 프로젝트 관리](https://koharu.rs/how-to/export-and-manage-projects/)를
참고하세요.

### MCP 서버

Koharu에는 로컬 에이전트 연동을 위한 MCP 서버가 내장되어 있습니다. 기본적으로 임의의
로컬 포트를 사용하며 `--port`로 고정할 수 있습니다.

```bash
# macOS / Linux
koharu --port 9999

# Windows
koharu.exe --port 9999
```

클라이언트의 MCP 주소를 `http://localhost:9999/mcp`로 설정하세요.

관련 문서:
[GUI, 헤드리스 및 MCP 모드](https://koharu.rs/how-to/run-gui-headless-and-mcp/),
[MCP 클라이언트 설정](https://koharu.rs/how-to/configure-mcp-clients/),
[MCP 도구 레퍼런스](https://koharu.rs/reference/mcp-tools/)

### 헤드리스 모드

데스크톱 창 없이 Koharu를 실행할 수도 있습니다.

```bash
# macOS / Linux
koharu --port 4000 --headless

# Windows
koharu.exe --port 4000 --headless
```

실행 후 `http://localhost:4000`에서 웹 클라이언트에 접속할 수 있습니다.

### 런타임 설정

공용 로컬 데이터 경로와 다운로드/프로바이더 요청에 사용하는 HTTP 연결 제한 시간,
읽기 제한 시간 및 재시도 횟수를 설정할 수 있습니다. 이 값들은 시작할 때 로드되므로
변경 사항을 저장하면 앱이 다시 시작됩니다.

### Google Fonts 및 사용자 글꼴

Koharu는 번역 텍스트 렌더링을 위한 Google Fonts를 기본 지원합니다. OpenType,
TrueType 및 가변 글꼴도 사용할 수 있습니다. 사용자 글꼴을 운영체제에 설치한 뒤
Koharu를 시작하면 시스템 글꼴 폴더를 검색하여 글꼴 선택 목록에 표시합니다.

### 텍스트 렌더링

Koharu의 전용 렌더러는 유니코드 인식 OpenType 셰이핑, 문자 체계별 줄바꿈, 정밀 글리프
메트릭 및 실제 글리프 경계를 사용합니다. 세로쓰기 CJK, 오른쪽에서 왼쪽으로 쓰는 문자,
글꼴 폴백, 세로 문장부호 정렬, 제한 영역 맞춤, 만화식 외곽선 및 효과 합성을 지원합니다.

## GPU 가속

원본 Koharu는 CUDA, 실험적 ZLUDA, Metal 및 Vulkan을 지원합니다. 이 AMD 포크는 주요
감지 및 인페인팅 엔진에 ncnn Vulkan 경로를 추가했습니다. 가속 백엔드를 사용할 수 없으면
CPU 폴백이 동작합니다.

### AMD ncnn Vulkan

Windows에서 최신 AMD 그래픽 드라이버와 Vulkan 런타임이 필요합니다. 이 포크에서 GPU
가속 로그는 다음 형태로 표시됩니다.

```text
using GPU backend backend="ncnn-vulkan" engine="speech-bubble-segmentation"
using GPU backend backend="ncnn-vulkan" engine="yuzumarker-font-detection"
using GPU backend backend="ncnn-vulkan" engine="lama-manga"
```

### CUDA

Windows와 Linux에서 NVIDIA GPU를 사용하는 전체 로컬 파이프라인을 지원합니다. 원본
배포판은 CUDA Toolkit 13.0 런타임을 포함하며 최초 실행 시 필요한 DLL을 앱 데이터
폴더에 추출합니다. NVIDIA 드라이버는 최신 버전을 권장하며, 지원 기준은 일반적으로
컴퓨트 성능 8.0 이상입니다.

### ZLUDA(실험적)

ZLUDA는 일부 CUDA 작업을 AMD GPU에서 실행하는 호환 계층입니다. Windows에서 사용하려면
[AMD HIP SDK](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)가 필요합니다.

### Metal

Apple Silicon Mac에서는 Metal을 지원하며 일반 설치 외에 별도 런타임 설정이 필요하지
않습니다.

### CPU 강제 사용

```bash
# macOS / Linux
koharu --cpu

# Windows
koharu.exe --cpu
```

## 머신러닝 모델

Koharu는 하나의 모델로 전체 페이지를 처리하는 대신 여러 비전 및 언어 모델을 단계별로
사용합니다.

### 감지 및 레이아웃

- [anime-text-yolo](https://huggingface.co/mayocream/anime-text-yolo): 텍스트 블록 감지
- [comic-text-bubble-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector): 텍스트 및 말풍선 공동 감지
- [comic-text-detector](https://huggingface.co/mayocream/comic-text-detector): 텍스트 분할 마스크
- [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors): 문서 레이아웃 분석
- [speech-bubble-segmentation](https://huggingface.co/mayocream/speech-bubble-segmentation): 말풍선 전용 감지

### OCR

- [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6): 다국어 OCR
- [Manga OCR](https://huggingface.co/mayocream/manga-ocr): 만화 OCR
- [MIT 48px OCR](https://huggingface.co/mayocream/mit48px-ocr): OCR

### 인페인팅

- [FLUX.2 Klein 4B](https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF): FLUX.2 기반 인페인팅
- [lama-manga](https://huggingface.co/mayocream/lama-manga): 만화 인페인팅
- [aot-inpainting](https://huggingface.co/mayocream/aot-inpainting): 인페인팅

### 글꼴 분석

- [YuzuMarker.FontDetection](https://huggingface.co/fffonion/yuzumarker-font-detection): 글꼴 및 색상 감지

필요한 원본 모델은 처음 사용할 때 자동으로 다운로드됩니다. 일부 모델은 upstream
Hugging Face 저장소에서 직접 사용하며, Rust용 safetensors 변환본은
[mayocream Hugging Face](https://huggingface.co/mayocream)에 제공됩니다.

## 대규모 언어 모델

Koharu는 로컬 및 원격 LLM 백엔드를 모두 지원합니다. 로컬 모델은
[llama.cpp](https://github.com/ggml-org/llama.cpp)를 통해 실행되고 필요할 때 다운로드됩니다.

### 범용 로컬 모델

- Gemma 4 instruct 계열
- Qwen 3.5 계열
- Qwen 3.6 계열

### 번역 특화 모델

- [vntl-llama3-8b-v2](https://huggingface.co/lmg-anon/vntl-llama3-8b-v2-gguf): 품질 중심 Q5_K_M 모델
- [lfm2.5-1.2b-instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF): 저사양용 소형 다국어 모델
- [Sugoi 14B/32B Ultra](https://huggingface.co/sugoitoolkit): 고용량 번역 모델
- [sakura-galtransl-7b-v3.7](https://huggingface.co/SakuraLLM/Sakura-GalTransl-7B-v3.7): 품질과 속도의 균형
- [sakura-1.5b-qwen2.5-v1.0](https://huggingface.co/shing3232/Sakura-1.5B-Qwen2.5-v1.0-GGUF-IMX): 경량 번역 모델
- [hunyuan-mt-7b](https://huggingface.co/Mungert/Hunyuan-MT-7B-GGUF): 다국어 번역 모델

메모리가 제한된 환경에서는 작은 모델부터 시작하세요. VRAM이나 RAM이 충분하면 일반적으로
7B 및 8B급 모델이 더 나은 번역 품질을 제공합니다.

### 클라우드 및 번역 프로바이더

로컬 모델 대신 OpenAI, Gemini, Claude 및 DeepSeek API를 사용할 수 있습니다. DeepL,
Google Cloud Translation 및 Caiyun 같은 기계 번역 서비스도 지원합니다. LM Studio,
OpenRouter 및 `/v1/models`, `/v1/chat/completions`를 제공하는 OpenAI 호환 서버도 연결할
수 있습니다.

API 키는 일반 텍스트 설정 파일이 아니라 시스템 키체인에 저장됩니다. 원격 프로바이더를
사용하면 번역 대상으로 선택된 OCR 텍스트가 해당 서비스로 전송된다는 점에 유의하세요.

### Codex 이미지 변환

현재 원본 페이지와 프롬프트를 Codex로 보내 이미지 전체 번역, 원문 제거 및 재생성을
한 번에 수행할 수 있습니다. ChatGPT Codex 사용 권한과 2단계 인증이 필요하며, 이미지는
ChatGPT Codex 백엔드에서 처리됩니다.

## 설치

원본 Koharu의 공식 바이너리는
[릴리스 페이지](https://github.com/mayocream/koharu/releases/latest)에서 받을 수 있습니다.
이 AMD 포크의 변경 사항을 사용하려면 현재 소스에서 직접 빌드해야 합니다.

### WinGet

```bash
winget install koharu
```

### Homebrew

```bash
brew install --cask koharu
```

### Docker

```bash
docker pull ghcr.io/mayocream/koharu:latest
docker run -p 4000:4000 --gpus all ghcr.io/mayocream/koharu:latest
```

위 설치 명령은 원본 공식 버전을 설치하며 AMD 패치 포크 빌드는 아닙니다.

## 문제 해결

자세한 로그와 시스템 정보를 출력하려면 디버그 모드를 사용하세요.

```bash
# macOS / Linux
koharu --debug

# Windows
koharu.exe --debug
```

PowerShell에서 더 상세한 로그를 활성화할 수 있습니다.

```powershell
$env:RUST_LOG="debug"
.\koharu.exe
```

Vulkan 엔진 대신 `No GPU support detected`가 표시되면 `KOHARU_NCNN_ROOT` 경로와
ncnn 아티팩트 파일을 확인하세요.

## 개발 및 빌드

### 필수 도구

- [Rust](https://www.rust-lang.org/tools/install) 1.95 이상(Rust 2024 edition)
- [Bun](https://bun.sh/) 1.0 이상
- [CMake](https://cmake.org/)
- [LLVM/Clang](https://llvm.org/) 15 이상
- Vulkan을 지원하는 최신 AMD 그래픽 드라이버

CUDA/ZLUDA 빌드에는 CUDA Toolkit 13.0 및 AMD HIP SDK가 추가로 필요할 수 있지만,
AMD Vulkan 빌드는 CUDA/NVCC 없이 구성할 수 있습니다.

### 의존성 설치

```bash
bun install
```

### 개발 모드

```bash
bun dev
```

### Windows AMD 빌드

```powershell
$env:LIBCLANG_PATH="C:\Program Files\LLVM\bin"
$env:KOHARU_NCNN_ROOT="$PWD\temp\vulkan-pilot"
bun run build:directml
```

빌드 결과는 `target/release/koharu.exe`에 생성됩니다.

## 후원

원본 Koharu가 유용하다면 프로젝트 제작자를 후원해 주세요.

- [GitHub Sponsors](https://github.com/sponsors/mayocream)
- [Patreon](https://www.patreon.com/mayocream)

## 기여자

Koharu를 발전시킨 모든 원본 프로젝트 기여자에게 감사드립니다.

<a href="https://github.com/mayocream/koharu/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=mayocream/koharu" alt="Koharu 기여자" />
</a>

## 라이선스

Koharu 및 이 포크의 수정 소스는 [GNU General Public License v3.0](LICENSE)에 따라
배포됩니다. 원본 프로젝트의 저작권과 기여 이력은 그대로 유지됩니다.
