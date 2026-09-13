# Contratos de datos crudos

FirstWatch conserva una representación canónica estable y deja las features exactas dentro de cada bundle. El cliente integra una sola vez su base bancaria con estos contratos; cada runner transforma el mismo histórico según su notebook.

## Solicitud de cuenta

`POST /api/applications` recibe:

```json
{
  "application_id": "application-001",
  "account_id": "account-001",
  "alias": "Demo applicant",
  "event_time": "2026-09-13T09:00:00Z",
  "raw_payload": {"source columns": "unchanged values"},
  "labels": {"fraud_bool": 0}
}
```

BAF aporta `raw_payload`. `fraud_bool` y cualquier target permanecen en `labels`, se almacenan para evaluación y son eliminados antes de inferencia. `event_time` representa cuándo el banco observó la solicitud, no una columna inventada de BAF.

## Evento transaccional

`POST /api/events` recibe:

```json
{
  "event_id": "transaction-001",
  "account_id": "account-001",
  "event_time": "2026-09-13T09:03:00Z",
  "type": "transfer",
  "amount": 125.50,
  "currency": "USD",
  "counterparty_id": "merchant-007",
  "description": "Dataset replay",
  "raw_payload": {"all non-label source columns": "preserved"},
  "labels": {"fraud_type": "card_testing"}
}
```

Tipos permitidos: `deposit`, `purchase`, `transfer`, `cash_out`, `withdrawal` y `payment`. Los montos son no negativos; el tipo expresa la dirección. Los timestamps requieren zona horaria y se envían en orden por cuenta. Una cuenta usa una moneda; cualquier conversión ocurre antes del contrato.

## Entradas de runners

El orquestador construye `{entity_id, as_of, application, events, context}`. Sólo se incluyen eventos con `event_time <= as_of`; las etiquetas se eliminan recursivamente. El pipeline de onboarding utiliza `application`. Los pipelines de vigilancia pueden usar el evento actual y el prefijo histórico, pero nunca filas futuras.

La base guarda tanto el snapshot crudo/derivado como el vector transformado, versión de preprocessing y hash del contrato. Esto permite auditar diferencias entre Colab y deployment sin crear una base distinta por modelo.
