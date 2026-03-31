# BlackOps Forensic Recovery

**Motor forense de grado militar para recuperación de video con ensamblaje no lineal.**

```
    ____  __            __        ____             _____                    
   / __ )/ /___  ____ _/ /__     / __ \____ ___   / __(_)_______  ____ _____ 
  / __  / / __ \/ __ `/ / _ \   / /_/ / __ `__ \ / /_/ / ___/ _ \/ __ `/ __ \
 / /_/ / / / / / /_/ / /  __/  / ____/ / / / / // __/ / /  /  __/ /_/ / / / /
/_____/_/_/ /_/\__, /_/\___/  /_/   /_/ /_/ /_//_/ /_/_/   \___/\__,_/_/ /_/ 
              /____/                                                          
```

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-zone)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Características

- **Ensamblaje No Lineal**: Reconstruye videos fragmentados usando teoría de grafos
- **Aceleración GPU**: Implementación CUDA nativa (sin Numba)
- **I/O Optimizado**: Buffers alineados, lectura asíncrona, I/O directo
- **Análisis de Entropía**: Detección de datos comprimidos/cifrados
- **Validación Forense**: Integración con FFprobe para verificación
- **Soporte Multi-Formato**: MP4, AVI, MPEG-TS, WMV, MKV, MOV, FLV

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                        BLACK OPS FORENSIC                        │
├─────────────────────────────────────────────────────────────────┤
│  FASE ALPHA                    │  FASE OMEGA                    │
│  ───────────                   │  ───────────                   │
│  ┌──────────────┐              │  ┌──────────────┐              │
│  │ DiskReader   │──┐           │  │ Graph        │              │
│  │ (I/O Opt.)   │  │           │  │ Assembler    │              │
│  └──────────────┘  │           │  │ (Cerebro)    │              │
│         │          │           │  └──────────────┘              │
│         ▼          │           │         │                      │
│  ┌──────────────┐  │           │         ▼                      │
│  │ CudaScanner  │  │           │  ┌──────────────┐              │
│  │ (GPU/CPU)    │◄─┘           │  │ TriageJudge  │              │
│  └──────────────┘              │  │ (FFprobe)    │              │
│         │                      │  └──────────────┘              │
│         ▼                      │         │                      │
│  ┌──────────────┐              │         ▼                      │
│  │ FragmentNode │─────────────►│  ┌──────────────┐              │
│  │ (Grafo)      │              │  │ Archivos     │              │
│  └──────────────┘              │  │ Recuperados  │              │
│                                │  └──────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

## Requisitos

### Sistema
- Windows 10/11 (x64) o Linux (x64)
- CUDA Toolkit 12.x (opcional, para aceleración GPU)
- CMake 3.20+
- Compilador C++20 (MSVC 2022, GCC 12+, Clang 15+)

### Hardware Recomendado
- CPU: 8+ cores (Ryzen 7 / Intel i7)
- RAM: 32GB+
- GPU: NVIDIA RTX 4060+ (opcional)
- Almacenamiento: SSD NVMe para archivo RAW

### Dependencias
- [spdlog](https://github.com/gabime/spdlog) - Logging
- [fmt](https://github.com/fmtlib/fmt) - Formatting
- [CLI11](https://github.com/CLIUtils/CLI11) - CLI
- [Catch2](https://github.com/catchorg/Catch2) - Tests (opcional)

## Compilación

### Windows (Visual Studio 2022)

```powershell
# Clonar repositorio
git clone https://github.com/blackops/forensic-recovery.git
cd forensic-recovery

# Crear directorio de build
mkdir build && cd build

# Configurar con CMake
cmake .. -G "Visual Studio 17 2022" -A x64 `
    -DENABLE_CUDA=ON `
    -DCMAKE_BUILD_TYPE=Release

# Compilar
cmake --build . --config Release --parallel
```

### Linux

```bash
# Instalar dependencias (Ubuntu/Debian)
sudo apt-get install cmake g++-12 cuda-toolkit-12

# Compilar
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
make -j$(nproc)
```

## Uso

### Comandos Básicos

```bash
# Escaneo completo
BlackOps_Forensic.exe -i imagen.raw -o ./recuperados

# Solo fase de detección (Alpha)
BlackOps_Forensic.exe -i imagen.raw --phase alpha

# Solo fase de ensamblaje (Omega) con fragmentos pre-detectados
BlackOps_Forensic.exe -i imagen.raw --phase omega

# Modo CPU (sin GPU)
BlackOps_Forensic.exe -i imagen.raw --force-cpu

# Filtros específicos
BlackOps_Forensic.exe -i imagen.raw --formats mp4,avi --min-size 10MB

# Salida verbose con reporte
BlackOps_Forensic.exe -i imagen.raw -V --report reporte.json
```

### Opciones

| Opción | Descripción | Default |
|--------|-------------|---------|
| `-i, --input` | Archivo RAW/IMG de entrada | *requerido* |
| `-o, --output` | Directorio de salida | `./recovered` |
| `--phase` | Fase: alpha, omega, full | `full` |
| `--mode` | Modo: auto, gpu, cpu | `auto` |
| `--chunk-size` | Tamaño de chunk en MB | `512` |
| `--min-confidence` | Confianza mínima (0-1) | `0.7` |
| `--formats` | Formatos objetivo | todos |
| `--ffprobe` | Ruta a ffprobe | `ffprobe` |
| `-V, --verbose` | Modo verbose | `false` |
| `--json` | Salida JSON | `false` |

## Algoritmos

### Detección de Fragmentos (Fase Alpha)

1. **Lectura por Chunks**: Archivos de 200GB+ se leen en chunks de 512MB
2. **Análisis de Entropía**: Shannon entropy para detectar datos comprimidos
3. **Detección de Firmas**: Magic numbers de contenedores de video
4. **Scoring**: Cada fragmento recibe un score de confianza

### Ensamblaje No Lineal (Fase Omega)

1. **Construcción del Grafo**: Nodos = fragmentos, Aristas = conexiones probables
2. **Cálculo de Pesos**: Basado en proximidad temporal, entropía, firmas
3. **Búsqueda de Caminos**: Algoritmos de máxima verosimilitud
4. **Validación**: FFprobe verifica la integridad de archivos ensamblados

## Rendimiento

| Hardware | Archivo | Velocidad |
|----------|---------|-----------|
| RTX 4060 + NVMe | 200GB RAW | ~2-3 GB/s |
| CPU 16 cores + NVMe | 200GB RAW | ~500-800 MB/s |
| CPU 8 cores + SSD | 200GB RAW | ~200-300 MB/s |

## Estructura del Proyecto

```
BlackOps_Forensic/
├── include/
│   ├── core/           # Tipos, FragmentNode, GraphAssembler, TriageVerdict
│   ├── io/             # DiskReader, AlignedBuffer
│   ├── gpu/            # CudaScanner, kernels CUDA
│   └── utils/          # Logger, Profiler
├── src/
│   ├── core/           # Implementaciones core
│   ├── io/             # Implementaciones I/O
│   ├── gpu/            # Kernels CUDA (.cu)
│   ├── utils/          # Utilidades
│   └── main.cpp        # CLI
├── tests/              # Tests unitarios
├── CMakeLists.txt      # Configuración CMake
└── README.md           # Este archivo
```

## Lecciones Aprendidas (Migración Python → C++)

### Problemas de Python Resueltos

| Problema | Solución en C++ |
|----------|-----------------|
| GIL (Global Interpreter Lock) | Paralelismo nativo con threads |
| Numba inestable | CUDA C++ nativo |
| Memory overhead de Python | `std::vector` pre-asignados, punteros crudos |
| `IndexError` en bounds | Iteradores seguros, span, verificaciones en debug |
| `FutureWarning` NumPy | Tipado fuerte con `<expected>` |
| Lentitud de I/O | I/O directo Windows, buffers alineados |

## Contribuir

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/nueva-feature`)
3. Commit tus cambios (`git commit -am 'Agrega nueva feature'`)
4. Push a la rama (`git push origin feature/nueva-feature`)
5. Abre un Pull Request

## Licencia

MIT License - Ver [LICENSE](LICENSE) para detalles.

## Contacto

- Issues: [GitHub Issues](https://github.com/blackops/forensic-recovery/issues)
- Email: forensic@blackops.dev

---

**Disclaimer**: Esta herramienta está diseñada para uso forense legítimo. El uso indebido es responsabilidad exclusiva del usuario.
