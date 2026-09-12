# FirstWatch

Sistema de detección temprana y monitoreo de cuentas potencialmente fraudulentas para neobancos. El producto combina evaluación de riesgo al abrir una cuenta, análisis de transacciones y detección de anomalías secuenciales para alertar a un analista antes de que ocurra un *cash-out* o *exit*.

> Proyecto de hackathon. Tiempo total de ejecución: 22 horas. La prioridad es entregar una demostración reproducible, explicable y técnicamente creíble; no una plataforma bancaria lista para producción.

## Plan ejecutivo

### Objetivo del MVP

Construir una aplicación web que permita:

1. Ingresar o reproducir una nueva solicitud de cuenta.
2. Calcular su riesgo inicial mediante el modelo de onboarding entrenado con BAF.
3. Simular cronológicamente la actividad de la cuenta.
4. Reevaluar su riesgo con un modelo transaccional y un LSTM-Autoencoder.
5. Combinar las señales mediante una política configurable.
6. Crear una alerta cuando el riesgo agregado supere el umbral elegido por el banco.
7. Mostrar al analista qué cambió, por qué se elevó el riesgo y qué eventos precedieron la alerta.

El umbral será ajustable para representar la tolerancia del cliente entre falsos positivos —fricción y pérdida de clientes legítimos— y falsos negativos —fraude no detectado—.

## Decisiones principales

- **Nessie API no forma parte del MVP.** No aporta las etiquetas, secuencias históricas ni variables necesarias para demostrar de forma controlada los modelos.
- **La aplicación no entrenará modelos.** Los responsables de ML entregarán notebooks terminados y archivos `.pickle` antes de la hora 11.
- **Cada modelo se ejecutará en un entorno aislado.** Esto evita que versiones incompatibles de Python, scikit-learn, XGBoost, LightGBM o TensorFlow bloqueen toda la aplicación.
- **La base de datos conservará datos crudos y snapshots de features.** No se fijará prematuramente un esquema rígido basado en variables que aún pueden cambiar.
- **La demo será una reproducción determinista.** Usará datos no vistos de BAF/MoMTSim y, cuando sea necesario, pequeñas modificaciones sintéticas explícitamente etiquetadas.
- **La comunicación con los modelos tendrá un contrato HTTP común.** El backend no necesitará conocer las particularidades internas de cada notebook.

## Arquitectura propuesta

```text
┌──────────────────────────────┐
│ Next.js                      │
│ Dashboard, timeline, alertas │
└──────────────┬───────────────┘
               │ REST + SSE
┌──────────────▼───────────────┐
│ FastAPI Orchestrator         │
│ Casos, replay, fusion, API   │
└───────┬──────────┬───────────┘
        │          │
        │          └────────────────────────┐
        │                                   │
┌───────▼────────┐   ┌──────────────────┐   ┌──────────────────┐
│ Onboarding     │   │ Transaction      │   │ Sequence         │
│ model runner   │   │ model runner     │   │ LSTM-AE runner   │
└───────┬────────┘   └────────┬─────────┘   └────────┬─────────┘
        └─────────────────────┴───────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ PostgreSQL        │
                    │ Raw + features +  │
                    │ scores + alerts   │
                    └───────────────────┘
```

### Componentes

- **Frontend:** dashboard para cartera, detalle de cuenta, línea de tiempo, controles de simulación y bandeja de alertas.
- **Orquestador:** recibe eventos, guarda datos, solicita inferencias, fusiona scores, aplica umbrales y publica actualizaciones.
- **Model runners:** un servicio aislado por modelo, con dependencias y preprocesamiento propios.
- **PostgreSQL:** fuente de verdad para cuentas, eventos, inferencias, alertas y ejecuciones de demo.
- **Scenario engine:** reproduce escenarios con reloj lógico y velocidad configurable.

## Estructura del repositorio

```text
.
├── README.md
├── docker-compose.yml
├── .env.example
├── apps/
│   ├── web/                         # Next.js
│   │   ├── app/
│   │   │   ├── dashboard/
│   │   │   ├── accounts/[id]/
│   │   │   └── alerts/
│   │   ├── components/
│   │   └── lib/api.ts
│   └── api/                         # FastAPI orquestador
│       ├── app/
│       │   ├── api/routes/
│       │   ├── core/
│       │   ├── db/
│       │   ├── services/
│       │   │   ├── feature_store.py
│       │   │   ├── model_gateway.py
│       │   │   ├── risk_fusion.py
│       │   │   ├── alert_service.py
│       │   │   └── scenario_engine.py
│       │   └── main.py
│       └── tests/
├── models/
│   ├── onboarding/
│   │   ├── Dockerfile
│   │   ├── requirements.lock
│   │   ├── manifest.json
│   │   ├── feature_contract.json
│   │   ├── adapter.py
│   │   ├── artifact/                # .pickle entregado; no versionar si es grande
│   │   └── tests/golden_cases.json
│   ├── transaction/
│   └── sequence/
├── configs/
│   ├── fusion_policy.json
│   ├── thresholds.json
│   └── demo_scenarios.yaml
├── data/
│   ├── raw/                         # Ignorado por Git
│   ├── curated/                     # Muestras mínimas para la demo
│   └── schemas/
├── scripts/
│   ├── inspect_artifact.py
│   ├── prepare_demo_data.py
│   ├── validate_model_bundle.py
│   └── seed_demo.py
├── db/
│   ├── migrations/
│   └── seed/
├── docs/
│   ├── architecture.md
│   ├── model-handoff-checklist.md
│   └── demo-script.md
└── EDA/                             # Análisis exploratorio existente
```

## Integración de modelos entregados

### Condición de entrega

Antes de la hora 11, cada responsable de ML entregará:

- notebook ejecutado de principio a fin;
- archivo `.pickle` descargado desde Colab;
- dataset y variable objetivo utilizados;
- orden final de features;
- explicación del split, semilla y preprocesamiento;
- ejemplo de una entrada válida y su salida esperada;
- versiones de Python y librerías, si están disponibles.

Como los notebooks y la aplicación pueden vivir en entornos distintos, **no se cargará un pickle desconocido dentro del backend principal**. Cada artefacto se inspeccionará y se encapsulará en su propio runner.

> Seguridad: Python advierte que `pickle` puede ejecutar código durante la deserialización. Solo se cargarán archivos creados por el equipo y cada carga se realizará dentro de un contenedor aislado.

### Proceso de recepción

1. Revisar imports, instalaciones, versión del kernel y celdas ejecutadas.
2. Identificar dataset, target, split, semilla y cualquier filtrado previo.
3. Enumerar features crudas, variables derivadas, orden de columnas y tipos.
4. Documentar imputación, codificación, escalamiento, transformación logarítmica y ventanas temporales.
5. Para el LSTM-AE, registrar longitud de secuencia, padding, máscara, orden temporal y definición del error de reconstrucción.
6. Localizar exactamente la celda que genera el `.pickle`.
7. Inspeccionar el artefacto de forma estática antes de cargarlo.
8. Recrear su entorno mínimo en un contenedor.
9. Construir un adaptador que reciba el contrato común y replique el preprocesamiento del notebook.
10. Comparar predicciones del runner contra ejemplos dorados obtenidos en el notebook.

### Semáforo de integración

| Estado | Condición | Acción |
|---|---|---|
| Verde | El pickle contiene pipeline completo y metadatos suficientes | Integración directa y prueba de paridad |
| Ámbar | Solo contiene el estimador, pero el notebook reproduce todo el preprocessing | Reconstruir adaptador, congelar dependencias y validar contra casos dorados |
| Rojo | Faltan transformadores, orden de features, pesos, categorías o estado aprendido | Devolver al responsable para reexportación; no adivinar parámetros |

### Formato de exportación preferido

Si todavía es posible modificar la exportación, cada equipo debe empaquetar modelo y preprocesamiento juntos:

```python
bundle = {
    "model": fitted_model,
    "preprocessor": fitted_preprocessor,
    "feature_names": feature_names,
    "model_version": "hackathon-v1",
    "library_versions": library_versions,
    "metadata": metadata,
}
```

Para modelos de scikit-learn, la primera opción es exportar un `Pipeline` ya ajustado. Para TensorFlow/Keras, es preferible guardar el modelo en su formato nativo y serializar por separado el scaler y el contrato de secuencia.

## Contrato común de los model runners

Cada runner expone:

- `GET /health`: confirma que el proceso y el artefacto están disponibles.
- `GET /metadata`: devuelve versión, features, ventana requerida y semántica del score.
- `POST /validate`: valida esquema y devuelve errores sin inferencia.
- `POST /predict`: genera el score y las evidencias disponibles.

Entrada normalizada:

```json
{
  "entity_id": "acct_001",
  "as_of": "2026-09-12T12:00:00Z",
  "application": {},
  "events": [],
  "context": {
    "scenario_id": "sleeper_bustout"
  }
}
```

Salida normalizada:

```json
{
  "model": "transaction",
  "model_version": "hackathon-v1",
  "risk_score": 0.82,
  "raw_score": 3.14,
  "status": "ok",
  "reason_codes": [
    {"code": "VELOCITY_SPIKE", "impact": 0.31}
  ],
  "feature_snapshot_id": "feat_123",
  "latency_ms": 18
}
```

Cada bundle debe contener, como mínimo:

```text
manifest.json
feature_contract.json
adapter.py
requirements.lock
artifact/
tests/golden_cases.json
```

`manifest.json` documentará la tarea, el dataset, la versión del modelo, la semántica y rango del score, el umbral sugerido y las dependencias. `feature_contract.json` documentará nombres, orden, tipos, valores nulos, categorías, unidades, ventanas y valores por defecto permitidos.

## Esquema flexible de datos

Las tablas propuestas son:

- `applications`: solicitud original, `raw_payload JSONB`, etiqueta solo para evaluación y timestamps.
- `accounts`: entidad monitoreada, estado y política de riesgo asignada.
- `ledger_events`: transacciones y eventos normalizados con `raw_payload JSONB`.
- `feature_snapshots`: vector exacto enviado a un modelo, versión del transformador y hash del contrato.
- `model_invocations`: modelo, versión, score crudo, score normalizado, latencia y estado.
- `risk_scores`: resultado fusionado y contribución de cada señal.
- `alerts`: umbral cruzado, severidad, estado y momento de creación.
- `analyst_reviews`: decisión humana, notas y disposición final.
- `model_bundles`: registro de artefactos, contratos y compatibilidad.
- `scenario_runs`: estado del replay, velocidad y reloj lógico.
- `entity_links`: relaciones sintéticas o reales como dispositivo, IP, beneficiario o identidad compartida.

Las etiquetas de fraude se almacenarán únicamente para evaluación y explicación posterior. Nunca serán features de inferencia.

## Prevención de training-serving skew

El principio central es: **la inferencia debe ejecutar exactamente las transformaciones aprendidas durante entrenamiento**.

- Una normalización logarítmica aplicada al entrenar también debe ejecutarse al inferir.
- Scalers, imputadores, encoders y vocabularios ajustados deben conservar su estado aprendido.
- El orden de columnas es parte del contrato, no un detalle de implementación.
- Las features de ventanas temporales se calculan únicamente con eventos disponibles hasta `as_of`.
- Ningún parámetro se reconstruye desde memoria o intuición.

Si el pipeline no fue serializado, solo se reconstruirá desde el notebook cuando el split y la semilla permitan reproducir los parámetros. La integración se acepta cuando los casos dorados producen scores iguales o dentro de una tolerancia numérica documentada.

## Fusión de riesgo

No se fijarán pesos definitivos hasta conocer la semántica real de los tres modelos. Cada adaptador convertirá su salida a un riesgo comparable entre 0 y 1:

- probabilidad calibrada: usar directamente después de verificar su clase positiva;
- margen o score no acotado: aplicar la transformación registrada en el notebook;
- error de reconstrucción: normalizar mediante la distribución de validación, por ejemplo con su función de distribución empírica.

Una política inicial, configurable en `configs/fusion_policy.json`, será:

```text
w_application = exp(-event_count / tau)
behavior_risk  = sum(quality_j * risk_j) / sum(quality_j)
core_risk      = w_application * application_risk
                 + (1 - w_application) * behavior_risk
```

Así, el riesgo de onboarding domina al inicio y pierde peso conforme aparece evidencia conductual. Los factores de calidad permiten ignorar una señal sin historial suficiente o un runner temporalmente degradado. La demo final debe operar con los tres modelos; el modo degradado existe solo para desarrollo y resiliencia.

## Datos y escenarios de demostración

### Fuente de datos

- **BAF:** solicitudes de cuenta para onboarding.
- **MoMTSim:** eventos y transacciones para comportamiento.
- **Datos sintéticos mínimos:** únicamente para crear relaciones demostrativas —por ejemplo, IP o dispositivo compartido— o controlar el momento de una anomalía. Todo campo sintético se etiquetará como tal.

El split real utilizado por cada notebook tiene prioridad. Si aún puede decidirse, la guía propuesta es:

- BAF: meses 0–4 entrenamiento, 5 validación, 6 prueba y 7 demo.
- MoMTSim: primeros 60 % entrenamiento, siguientes 20 % validación, siguientes 10 % prueba y últimos 10 % demo, siempre respetando orden temporal.

Los casos de demo no pueden haber participado en ajuste, calibración ni selección de umbral.

### Escenarios

1. **Cliente legítimo:** comportamiento estable y score bajo.
2. **Granja de identidades:** varias cuentas comparten señales sintéticas de dispositivo o IP; demuestra el filtro inicial y los vínculos.
3. **Sleeper / bust-out:** actividad inicial normal seguida de incremento de velocidad, montos y salida de fondos; demuestra el monitoreo dinámico.
4. **Falso positivo desafiante:** cuenta con señales atípicas pero justificación plausible; demuestra el valor del umbral y la revisión humana.

La selección final se hará después de integrar los modelos, porque la duración mínima y las features disponibles dependen de sus contratos. La curación de casos sirve para contar la historia; las métricas se reportan por separado sobre conjuntos completos y no vistos.

## Timelapse y experiencia de demo

Cada escenario será una lista inmutable de eventos con:

- `event_time`: momento simulado del evento;
- `observed_at`: momento en que el sistema lo procesa;
- `sequence_number`: orden estable para reproducibilidad.

Controles de la interfaz:

- iniciar y pausar;
- avanzar un evento;
- velocidades 1×, 5× y 20×;
- reiniciar con el mismo seed;
- cambiar el umbral del banco.

En cada paso se mostrarán el evento, scores individuales, score fusionado, umbral, tendencia y razones disponibles. La interfaz no codificará explicaciones falsas: mostrará `reason_codes` devueltos por los runners o, cuando no existan, cambios observables de features claramente marcados como descripción y no como causalidad del modelo.

### Endpoints del orquestador

```text
POST /api/scenarios/{id}/start
POST /api/scenarios/{id}/pause
POST /api/scenarios/{id}/step
POST /api/scenarios/{id}/reset
PATCH /api/scenarios/{id}/speed
GET /api/scenarios/{id}/stream       # Server-Sent Events
GET /api/accounts
GET /api/accounts/{id}
GET /api/accounts/{id}/timeline
GET /api/alerts
PATCH /api/alerts/{id}
PATCH /api/policies/{id}/threshold
```

## Stack recomendado

| Capa | Tecnología | Motivo |
|---|---|---|
| Frontend | Next.js + TypeScript + Tailwind CSS | Dashboard rápido, tipado y buena experiencia de demo |
| Gráficas | Recharts | Curva de riesgo y actividad con baja fricción |
| Backend | FastAPI + Pydantic | Contratos claros, rapidez de desarrollo y compatibilidad con Python/ML |
| Persistencia | PostgreSQL + SQLAlchemy/Alembic | JSONB flexible, auditoría y migraciones |
| Actualizaciones | Server-Sent Events | Suficiente para un flujo unidireccional de timelapse |
| Model serving | FastAPI por runner | Aislamiento y contrato uniforme |
| Contenedores | Docker Compose | Entornos reproducibles sin montar Kubernetes |
| Pruebas | Pytest + pruebas de contrato | Paridad, validación de bundles y flujo extremo a extremo |

No se propone Kafka, Kubernetes, un feature store externo ni microservicios adicionales durante el hackathon. La separación de runners responde a una incompatibilidad real de artefactos; el resto permanece como un monolito modular.

## Cronograma de 22 horas

El entrenamiento ocurre en paralelo a cargo de los responsables de ML y no forma parte de este plan de implementación.

### Horas 0–2: contrato y esqueleto

- Congelar el contrato común de inferencia y el formato de bundle.
- Crear Docker Compose, backend, frontend y PostgreSQL.
- Definir esquema mínimo y datos ficticios temporales.
- Entregar checklist de exportación a quienes entrenan los modelos.

### Horas 2–6: flujo vertical con stubs

- Implementar cuentas, eventos, scores y alertas.
- Crear runners simulados que respeten el contrato final.
- Conectar el dashboard al backend.
- Mostrar una cuenta, su timeline y una alerta extremo a extremo.

### Horas 6–10: motor de escenarios y UX

- Preparar replays deterministas.
- Implementar start, pause, step, reset y velocidades.
- Añadir ajuste de umbral y curva de riesgo.
- Preparar vistas de cartera y revisión humana.

### Horas 10–11: cierre previo a modelos

- Ejecutar pruebas de contrato contra stubs.
- Congelar cambios de interfaz.
- Preparar contenedores vacíos para los tres runners.
- Confirmar que cada equipo entrega notebook, pickle y casos esperados.

### Horas 11–15: recepción e integración

- Auditar los tres notebooks y clasificar cada artefacto verde/ámbar/rojo.
- Construir dependencias aisladas y adaptadores.
- Replicar preprocessing y validar casos dorados.
- Registrar metadata y feature contracts.

### Horas 15–18: fusión y datos definitivos

- Normalizar scores y ajustar la política de fusión.
- Seleccionar escenarios compatibles con las ventanas requeridas.
- Preparar muestras BAF/MoMTSim no vistas.
- Sustituir stubs por modelos reales.

### Horas 18–20: validación y narrativa

- Ejecutar pruebas extremo a extremo con los tres modelos.
- Verificar determinismo, alertas y modo degradado.
- Afinar etiquetas, razones y explicación del umbral.
- Ensayar el recorrido completo.

### Horas 20–22: estabilización y entrega

- Corregir únicamente fallas de severidad alta.
- Congelar código y datos de demo.
- Ejecutar un arranque limpio con Docker Compose.
- Grabar respaldo en video y realizar ensayo final.

## Distribución sugerida del equipo

| Persona | Responsabilidad principal |
|---|---|
| 1 | Backend, PostgreSQL, contrato y orquestación |
| 2 | Frontend, visualización y experiencia del analista |
| 3 | Motor de escenarios, datos de demo y pruebas E2E |
| 4 | Recepción de notebooks, runners, paridad y fusión |

Durante las horas 11–15, quien termine antes apoya la integración de modelos; ese es el punto de mayor riesgo del proyecto.

## Criterios de aceptación

- La aplicación arranca desde cero mediante una instrucción reproducible.
- Los tres runners responden a `/health`, `/metadata`, `/validate` y `/predict`.
- Cada modelo pasa al menos dos casos dorados contra su notebook.
- Una cuenta legítima completa el replay sin alerta injustificada en el escenario preparado.
- El escenario bust-out cruza el umbral antes o durante el evento crítico previsto.
- Cambiar el umbral modifica de forma visible la sensibilidad de alertas.
- Cada inferencia conserva payload crudo, snapshot de features, versión y score.
- El timelapse puede pausarse, avanzar, acelerarse y reiniciarse de forma determinista.
- Una caída de un runner se muestra como señal no disponible y no derriba toda la aplicación.
- La demo completa dura menos de cinco minutos y existe un video de respaldo.

## Implicaciones para una implementación real

El MVP demuestra arquitectura y flujo de decisión, pero una implantación bancaria requeriría además:

- mapear el esquema real del cliente al contrato canónico;
- ejecutar pipelines de features versionados tanto en entrenamiento como en inferencia;
- sustituir cargas manuales por eventos o APIs autenticadas;
- cifrado, secretos administrados, RBAC, auditoría inmutable y retención definida;
- monitoreo de drift, calidad de datos, latencia y calibración;
- pruebas de sesgo, explicabilidad, proceso de apelación y gobierno de modelos;
- estrategia de champion/challenger, rollback y reentrenamiento;
- pruebas de carga, alta disponibilidad y objetivos de servicio.

Ante preguntas del cliente, la respuesta clave es que FirstWatch no exige que el banco entregue datos ya normalizados: el banco entrega datos bajo un contrato acordado, el sistema valida y transforma esos datos con la misma versión del pipeline usada en entrenamiento, y registra cada transformación para auditoría.

## Supuestos y límites

- El equipo de ML entrega los tres artefactos antes de la hora 11.
- Los archivos provienen de personas de confianza del equipo.
- Se dispone de suficientes filas no vistas para seleccionar cuatro escenarios.
- La interfaz y el pitch se presentarán en inglés, aunque este plan se mantiene en español para coordinación interna.
- Las métricas de la demo son ilustrativas hasta validar los modelos entregados.
- El producto recomienda revisión humana; no bloquea o cierra cuentas automáticamente.

## Referencias técnicas

- [Bank Account Fraud Dataset Suite (BAF)](https://github.com/feedzai/bank-account-fraud)
- [MoMTSim: Mobile Money Transaction Simulator](https://pmc.ncbi.nlm.nih.gov/articles/PMC12036017/)
- [Persistencia de modelos en scikit-learn](https://scikit-learn.org/stable/model_persistence.html)
- [Advertencia de seguridad de `pickle`](https://docs.python.org/3/library/pickle.html)
