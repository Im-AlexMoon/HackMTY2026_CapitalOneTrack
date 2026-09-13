# Checklist de preparación e incorporación de modelos

Estado al 13 de septiembre de 2026. Un elemento marcado como listo tiene implementación y prueba automatizada; los elementos abiertos son validaciones científicas o de datos que no deben inferirse.

## Plataforma e integración — listo

- [x] PostgreSQL almacena solicitudes, eventos, features, invocaciones, scores, alertas y revisiones.
- [x] API canónica separa `raw_payload` de `labels` y rechaza tiempos sin zona, montos inválidos, duplicados conflictivos y eventos desordenados.
- [x] Tres runners aislados comparten `/health`, `/metadata`, `/validate` y `/predict`.
- [x] Un runner no preparado devuelve `risk_score: null`; nunca se interpreta como riesgo cero.
- [x] Artefactos sólo se cargan con confianza explícita, hash correcto, entorno exacto y verificación dorada vigente.
- [x] `/api/models/status` y el dashboard distinguen servicio disponible de bundle listo para uso externo.
- [x] La demo sintética permanece identificada y separada del modo de modelos externos.

## Datos — herramientas listas

- [x] `profile_dataset.py` perfila CSV en streaming sin guardar valores de filas.
- [x] `init_transaction_mapping.py` genera un borrador de mapping desde los encabezados reales y exige revisión humana.
- [x] `prepare_demo_data.py` registra SHA-256/procedencia, separa etiquetas y puede seleccionar historias completas por cuenta.
- [x] `build_demo_package.py` crea enlaces sintéticos explícitos entre solicitudes BAF e historiales transaccionales no relacionados.
- [x] Los contratos crudos de aplicación y evento están documentados en `raw-data-contracts.md`.
- [ ] Confirmar el archivo exacto, versión y URL usada por los dos notebooks transaccionales.
- [ ] Ejecutar el generador de mapping y revisar todas las columnas, tipos de operación, moneda, timestamps y etiquetas.
- [ ] Obtener del equipo los identificadores reales de train/validation/test antes de marcar cualquier fila como holdout.
- [ ] Seleccionar las cuentas finales de demostración y completar `configs/datasets/demo-links.template.json`.

## Entrega ML — integración local completada

Para **onboarding**, **transaction** y **sequence**:

- [x] Copiar y verificar los artefactos con `scripts/stage_model_artifacts.py`.
- [x] Completar dataset, target, versiones, hashes, formato, método de score y normalización en `manifest.json`.
- [x] Completar orden, tipos, nulabilidad y categorías en `feature_contract.json`.
- [x] Implementar `pipeline.py` sin `fit`, `fit_transform` ni acceso a eventos futuros.
- [x] Añadir dos baselines de integración por modelo en `tests/golden_cases.json`.
- [x] Para secuencia: registrar ventana de 10 eventos, sin padding, 33 features y ECDF de validación.
- [x] Ejecutar los casos en cada contenedor exacto y generar `verification.json` ligado al contenido.
- [ ] Confirmar los baselines con salidas independientes de Colab de los responsables de ML.

El equipo de ML debe seguir `model-owner-handoff.md`. El estado agregado se consulta sin cargar pickle:

```powershell
.\.venv\Scripts\python.exe scripts/model_readiness.py
```

## Puerta de activación

- [x] `scripts/model_readiness.py --require-ready` termina con código 0 en el entorno construido.
- [x] Los tres servicios muestran `ready_for_external: true` en `/api/models/status`.
- [x] Pruebas Python, frontend, build y Playwright están en verde.
- [x] El modo externo acepta una aplicación y diez eventos de principio a fin con tres inferencias `ok`.
- [ ] Recalibrar fusión y umbrales únicamente con validación; no presentar el score como probabilidad sin calibración.

No se debe cerrar ningún punto abierto mediante supuestos. Un cambio en artefacto, pipeline, contrato, dependencias o casos dorados invalida la verificación anterior.
