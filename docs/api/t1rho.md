# T1rho Models

T1rho spin-lock relaxometry models.

## T1Rho

::: qmrpy.models.t1rho.T1Rho

## Usage

```python
from qmrpy.models import T1Rho

model = T1Rho(tsl_ms=[0, 10, 30, 60])
signal = model.forward(m0=1000, t1rho_ms=70)
fit = model.fit(signal)
print(fit["t1rho_ms"])
```
