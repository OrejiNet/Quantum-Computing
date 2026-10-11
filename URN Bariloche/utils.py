"""
utils.py — helpers compartidos por los notebooks del curso ECCI 2026.

Nada de lo que hay acá manda trabajos al hardware real. Las funciones de
envío arman el registro y escriben el job_id a disco; el `.run()` lo dispara
el usuario desde la celda del notebook, con su flag ENVIAR en True.
"""

from __future__ import annotations

import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Estética común a todos los notebooks
# --------------------------------------------------------------------------

COLOR_IDEAL = "#2E5EAA"     # simulador sin ruido
COLOR_RUIDO = "#D1495B"     # simulador con modelo de ruido (FakeFez)
COLOR_HARD = "#0F7173"      # hardware real
COLOR_TEORIA = "#222222"    # curva analítica

PALETA = [COLOR_IDEAL, COLOR_RUIDO, COLOR_HARD, "#EDAE49", "#8367C7", "#66A182"]


def estilo():
    """Aplica el estilo de figuras del curso. Llamar una vez por notebook."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
    })


# --------------------------------------------------------------------------
# Conteos: normalización y barras de error
# --------------------------------------------------------------------------

def a_probabilidades(counts: dict, claves=None) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Convierte un dict de cuentas en (claves, probabilidades, error estándar).

    El error es el binomial sqrt(p(1-p)/N): es la barra que hay que dibujar
    cuando se compara un histograma contra una predicción teórica.
    """
    n = sum(counts.values())
    if claves is None:
        claves = sorted(counts)
    p = np.array([counts.get(k, 0) / n for k in claves])
    err = np.sqrt(p * (1 - p) / n)
    return list(claves), p, err


def claves_completas(n_bits: int) -> list[str]:
    """Todas las cadenas de n bits, en orden. Sirve para que los histogramas
    muestren también los resultados que salieron cero veces."""
    return [format(i, f"0{n_bits}b") for i in range(2 ** n_bits)]


def comparar_conteos(series: dict, titulo="", claves=None, teoria=None,
                     ax=None, ylabel="probabilidad"):
    """Barras agrupadas para comparar ideal / ruidoso / hardware sobre el
    mismo eje, con barras de error binomiales y, opcionalmente, la predicción
    teórica como marcas horizontales.

    series: {"etiqueta": counts_dict, ...}
    teoria: dict {clave: probabilidad} o None
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(max(5, 1.1 * len(claves or [1] * 8)), 3.4))
    if claves is None:
        claves = sorted({k for c in series.values() for k in c})
    x = np.arange(len(claves))
    m = len(series)
    ancho = 0.8 / m
    for i, (etiqueta, counts) in enumerate(series.items()):
        _, p, err = a_probabilidades(counts, claves)
        ax.bar(x + (i - (m - 1) / 2) * ancho, p, ancho, yerr=err, capsize=2,
               label=etiqueta, color=PALETA[i % len(PALETA)],
               error_kw={"lw": 0.8, "ecolor": "0.3"})
    if teoria is not None:
        for j, k in enumerate(claves):
            if k in teoria:
                ax.hlines(teoria[k], j - 0.45, j + 0.45, color=COLOR_TEORIA,
                          lw=1.6, linestyle="--",
                          label="teoría" if j == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(claves, rotation=90 if len(claves) > 8 else 0,
                       family="monospace")
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.legend()
    return ax


# --------------------------------------------------------------------------
# Simulación
# --------------------------------------------------------------------------

def correr_ideal(circuitos, shots=4096):
    """AerSimulator sin ruido. Devuelve counts (o lista de counts)."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    uno = not isinstance(circuitos, (list, tuple))
    lista = [circuitos] if uno else list(circuitos)
    res = sim.run(transpile(lista, sim), shots=shots).result()
    salida = [res.get_counts(i) for i in range(len(lista))]
    return salida[0] if uno else salida


_SIM_CACHE = {}


def correr_ruidoso(circuitos, backend_falso=None, shots=4096, opt=1):
    """AerSimulator con el modelo de ruido y el coupling map de un backend
    falso de IBM (por defecto FakeFez). Es lo más parecido al hardware que
    se puede conseguir sin gastar un segundo de QPU.

    OJO con qué es FakeFez: son datos REALES de ibm_fez (errores por qubit,
    T1/T2, duraciones, coupling map) pero de un SNAPSHOT congelado que viene
    dentro del paquete, calibrado el 2025-02-26. No es el chip de hoy, y no
    hace ninguna llamada a la red. Los resultados con ruido de estos notebooks
    son "cómo andaba ibm_fez ese día", no "cómo anda ahora".

    El simulador y el pass manager se CACHEAN: construir el modelo de ruido de
    un chip de 156 qubits tarda del orden de 20 s, y sin caché un notebook que
    llame a esta función cien veces se pasa media hora armando lo mismo.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeFez

    if backend_falso is None:
        backend_falso = FakeFez()
    clave = (backend_falso.name, opt)
    if clave not in _SIM_CACHE:
        _SIM_CACHE[clave] = (
            AerSimulator.from_backend(backend_falso),
            generate_preset_pass_manager(optimization_level=opt, backend=backend_falso))
    sim, pm = _SIM_CACHE[clave]
    uno = not isinstance(circuitos, (list, tuple))
    lista = [circuitos] if uno else list(circuitos)
    res = sim.run(pm.run(lista), shots=shots).result()
    salida = [res.get_counts(i) for i in range(len(lista))]
    return salida[0] if uno else salida


# --------------------------------------------------------------------------
# Costo de QPU: estimar antes de gastar
# --------------------------------------------------------------------------

def informe_transpilacion(circuitos, shots=1024, rep_delay=250e-6, overhead=2.0,
                          nombre_backend="") -> float:
    """Imprime profundidad, compuertas de dos qubits y costo estimado de QPU.

    El tiempo de QPU lo domina el ciclo de repetición por shot (~250 us en
    Heron), no la duración del circuito en sí: un circuito de profundidad 20
    dura ~2 us. Por eso la estimación es, esencialmente,
    n_circuitos x shots x rep_delay, más un overhead fijo por job.

    Es una estimación de orden de magnitud. El número real lo dice la consola
    de IBM Quantum al terminar el job, y ese es el que manda. (Contrastado el
    17/09/2026: 2 circuitos x 1024 shots en ibm_fez estimaban ~2.5 s y
    consumieron 3 s.)

    Los qubits ociosos del chip no cuestan tiempo: el ciclo de repeticion corre
    igual con 1 qubit activo que con 156. Transpilar tampoco cuesta nada, se
    puede repetir todas las veces que haga falta.
    """
    lista = circuitos if isinstance(circuitos, (list, tuple)) else [circuitos]
    print(f"backend        : {nombre_backend or '(sin fijar)'}")
    print(f"circuitos (PUB): {len(lista)}")
    print(f"shots c/u      : {shots}")
    print(f"{'#':>3}  {'profundidad':>11}  {'2q':>4}  {'activos':>7}  qubits físicos")
    total_2q = 0
    for i, c in enumerate(lista):
        n2 = sum(v for k, v in c.count_ops().items()
                 if k in ("cz", "ecr", "cx", "cy", "swap", "rzz"))
        total_2q += n2
        # OJO: c.num_qubits es el ancho del CHIP (156 en Heron), no lo que se
        # usa. Los qubits que realmente hacen algo son los del layout.
        activos = {q for inst in c.data for q in inst.qubits}
        try:
            fisicos = c.layout.final_index_layout()
            etiqueta = str(fisicos) if len(fisicos) <= 8 else f"{len(fisicos)} qubits"
        except (AttributeError, TypeError):
            etiqueta = "(sin transpilar)"
        print(f"{i:>3}  {c.depth():>11}  {n2:>4}  {len(activos):>7}  {etiqueta}")
    seg = len(lista) * shots * rep_delay + overhead
    print(f"\ncompuertas 2q totales : {total_2q}")
    print(f"costo estimado de QPU : ~{seg:.1f} s  "
          f"({seg / 600 * 100:.1f} % de los 10 min de la ventana de 28 dias)")
    return seg


# --------------------------------------------------------------------------
# Registro de jobs: un job_id perdido es tiempo de QPU perdido
# --------------------------------------------------------------------------

ARCHIVO_JOBS = "jobs.jsonl"


def huella_circuitos(circuitos) -> list[dict]:
    """Resumen de qué se mandó realmente: qubits activos, profundidad y ops.

    Existe por una razón concreta: los notebooks se editan, y meses después uno
    compara los datos de un job viejo contra la simulación del circuito NUEVO
    sin darse cuenta de que ya no son el mismo circuito. Guardar la huella hace
    que esa discrepancia se vea en vez de pasar desapercibida.
    """
    lista = circuitos if isinstance(circuitos, (list, tuple)) else [circuitos]
    salida = []
    for c in lista:
        activos = len({q for inst in c.data for q in inst.qubits})
        salida.append({"qubits_activos": activos, "profundidad": c.depth(),
                       "ops": {k: int(v) for k, v in c.count_ops().items()}})
    return salida


def registrar_job(job, backend_name: str, descripcion: str,
                  archivo: str = ARCHIVO_JOBS, circuitos=None, **extra) -> str:
    """Escribe el job_id a disco inmediatamente después de enviarlo.

    Se llama DENTRO del `if ENVIAR:` y en la misma celda que el .run().
    Si le pasás `circuitos`, guarda además su huella (ver huella_circuitos).
    """
    jid = job.job_id()
    if circuitos is not None:
        extra["huella"] = huella_circuitos(circuitos)
    registro = {
        "job_id": jid,
        "backend": backend_name,
        "descripcion": descripcion,
        "fecha": datetime.datetime.now().isoformat(timespec="seconds"),
        **extra,
    }
    with open(archivo, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    print("job_id:", jid)
    print("registrado en", os.path.abspath(archivo))
    return jid


def listar_jobs(archivo: str = ARCHIVO_JOBS, filtro: str = "") -> list[dict]:
    """Lee jobs.jsonl y lista lo enviado. `filtro` matchea en la descripción."""
    if not os.path.exists(archivo):
        print(f"todavía no existe {archivo}: no se envió nada.")
        return []
    registros = [json.loads(l) for l in open(archivo, encoding="utf-8") if l.strip()]
    if filtro:
        registros = [r for r in registros if filtro.lower() in r["descripcion"].lower()]
    for r in registros:
        print(f"{r['fecha']}  {r['backend']:<16} {r['job_id']}  {r['descripcion']}")
    return registros


def recuperar(service, job_id: str):
    """Trae un job por id. Nunca esperes la cola en vivo: mandá, guardá el id,
    y volvé más tarde con esto."""
    job = service.job(job_id)
    print("estado:", job.status())
    return job


def counts_de_resultado(resultado, i: int = 0, registro: str = None) -> dict:
    """Extrae counts del PrimitiveResult de SamplerV2, sin tener que acordarse
    del nombre del registro clásico.

    El DataBin nombra los campos según los registros clásicos del circuito: 'c'
    si lo creaste con QuantumCircuit(n, m), 'meas' si usaste measure_all().
    `.keys()` los lista, así que no hay que adivinar.
    """
    datos = resultado[i].data
    nombres = list(datos.keys())
    if registro is None:
        registro = nombres[0]
    if len(nombres) > 1:
        print(f"ojo: el circuito {i} tiene varios registros {nombres}, uso '{registro}'")
    return getattr(datos, registro).get_counts()


def ultimo_job(experimento: str = None, notebook: str = None,
               archivo: str = ARCHIVO_JOBS) -> dict | None:
    """El registro más reciente de jobs.jsonl que matchea. Evita tener que
    copiar y pegar job_ids a mano."""
    if not os.path.exists(archivo):
        return None
    regs = [json.loads(l) for l in open(archivo, encoding="utf-8") if l.strip()]
    if experimento:
        regs = [r for r in regs if r.get("experimento") == experimento]
    if notebook:
        regs = [r for r in regs if r.get("notebook") == notebook]
    return regs[-1] if regs else None


# --------------------------------------------------------------------------
# Post-selección: el equivalente a tapar una salida del imán con plomo
# --------------------------------------------------------------------------

def bit_de(cadena: str, i: int) -> str:
    """El bit clásico i de una cadena de counts.

    Qiskit es little-endian: en 'c2c1c0' el bit 0 es el de la DERECHA.
    """
    return cadena[::-1].replace(" ", "")[i]


def filtrar(counts: dict, condicion: dict, mantener=None) -> dict:
    """Se queda con los shots que cumplen `condicion` y devuelve la
    distribución de los bits de `mantener`.

    condicion : {indice_de_bit: '0' o '1'}   -- el haz que dejamos pasar
    mantener  : lista de indices de bits a reportar (default: los no fijados)

    Ejemplo: filtrar(counts, {0: '0'}, mantener=[1]) es "de los átomos que
    salieron arriba del primer imán, ¿qué dio el segundo?".
    """
    limpio = {k.replace(" ", ""): v for k, v in counts.items()}
    n = len(next(iter(limpio)))
    if mantener is None:
        mantener = sorted(set(range(n)) - set(condicion), reverse=True)
    salida = {}
    for s, c in limpio.items():
        if all(bit_de(s, i) == v for i, v in condicion.items()):
            clave = "".join(bit_de(s, i) for i in mantener)
            salida[clave] = salida.get(clave, 0) + c
    return salida


def supervivencia(counts: dict, condicion: dict) -> tuple[int, float]:
    """Cuántos shots sobreviven al filtro y qué fracción del total son.
    La intensidad del haz después de cada imán."""
    total = sum(counts.values())
    vivos = sum(filtrar(counts, condicion).values())
    return vivos, vivos / total if total else 0.0


def sumar_counts(lista) -> dict:
    """Junta varios dicts de counts en uno. Para construir un ensamble a
    partir de muchas preparaciones distintas."""
    total = {}
    for c in lista:
        for k, v in c.items():
            total[k] = total.get(k, 0) + v
    return total


def bloch_de_counts(counts_xyz, shots=None) -> np.ndarray:
    """Vector de Bloch a partir de tres histogramas medidos en X, Y y Z.
    Cada counts debe tener claves '0'/'1'. Acepta ensambles de distinto
    tamaño (post-seleccionados), normalizando cada uno por separado."""
    r = []
    for c in counts_xyz:
        n = shots or sum(c.values())
        r.append((c.get("0", 0) - c.get("1", 0)) / n if n else 0.0)
    return np.array(r)


def layout(fig, **kwargs):
    """fig.tight_layout() con red de contención.

    Las esferas de Bloch de Qiskit dibujan las flechas con FancyArrowPatch, y
    cuando una figura tiene *solo* ejes 3D el cálculo de bounding boxes de
    matplotlib falla con "'NoneType' object has no attribute 'get_extents'".
    En ese caso caemos a un subplots_adjust razonable.
    """
    try:
        fig.tight_layout(**kwargs)
    except AttributeError:
        fig.subplots_adjust(left=0.03, right=0.97, top=0.82, bottom=0.05,
                            wspace=0.05, hspace=0.20)


# --------------------------------------------------------------------------
# Interactividad (ipywidgets): siempre con figura estática de respaldo
# --------------------------------------------------------------------------

def _hay_widgets() -> bool:
    try:
        import ipywidgets  # noqa: F401
        return True
    except ImportError:
        return False


def postseleccion_interactiva(series: dict, nombres: dict = None, titulo: str = "",
                              teoria: dict = None, inicial: dict = None,
                              etiquetas_valores=("↑ (0)", "↓ (1)")):
    """Histograma con post-selección elegible con el mouse.

    series  : {"etiqueta": counts}   una o varias corridas sobre el mismo circuito
    nombres : {indice_de_bit: "imán 1"}  cómo se llama cada medición
    inicial : {indice_de_bit: "0"} filtro de la figura ESTÁTICA que se dibuja
              primero, para que el notebook guardado muestre algo sin kernel
    teoria  : {clave: probabilidad} marcas horizontales de la predicción

    Cada dropdown tapa o destapa una salida del aparato. Los bits que queden en
    "todos" son los que se grafican.
    """
    import matplotlib.pyplot as plt

    primera = next(iter(series.values()))
    n_bits = len(next(iter(primera)).replace(" ", ""))
    nombres = nombres or {i: f"bit {i}" for i in range(n_bits)}

    def dibujar(**seleccion):
        cond = {int(k[1:]): v for k, v in seleccion.items() if v != "todos"}
        libres = [i for i in range(n_bits) if i not in cond][::-1]
        if not libres:
            print("dejá al menos un bit en 'todos' para tener algo que graficar")
            return
        filtradas, vivos = {}, 0
        for etiqueta, counts in series.items():
            filtradas[etiqueta] = filtrar(counts, cond, mantener=libres)
            vivos = max(vivos, sum(filtradas[etiqueta].values()))
        total = sum(primera.values())
        detalle = ", ".join(f"{nombres.get(i, i)}={'↑' if v == '0' else '↓'}"
                            for i, v in sorted(cond.items())) or "sin filtrar"
        fig, ax = plt.subplots(figsize=(max(5.5, 1.0 * 2 ** len(libres)), 3.5))
        comparar_conteos(filtradas, claves=claves_completas(len(libres)),
                         ax=ax, teoria=teoria,
                         titulo=f"{titulo}\n{detalle} — {vivos}/{total} shots "
                               f"({vivos / total:.0%} del haz)")
        ax.set_xlabel("".join(f"$c_{i}$" for i in libres))
        plt.show()

    # 1) figura estática: sobrevive al guardado, se ve sin kernel
    dibujar(**{f"b{i}": (inicial or {}).get(i, "todos") for i in range(n_bits)})

    # 2) y encima, los controles
    if not _hay_widgets():
        print("(ipywidgets no está instalado: figura estática. "
              "conda install -n quantum -c conda-forge ipywidgets)")
        return None
    import ipywidgets as w
    from IPython.display import display

    controles = {
        f"b{i}": w.Dropdown(
            options=[("todos", "todos"), (etiquetas_valores[0], "0"),
                     (etiquetas_valores[1], "1")],
            value=(inicial or {}).get(i, "todos"),
            description=str(nombres.get(i, f"bit {i}")),
            style={"description_width": "initial"},
            layout=w.Layout(width="230px"))
        for i in range(n_bits)
    }
    salida = w.interactive_output(dibujar, controles)
    display(w.HBox(list(controles.values())), salida)
    return salida


def deslizador(fn, **rangos):
    """interact() con respaldo estático. `rangos` acepta lo mismo que ipywidgets:
    (min, max, paso) para números, listas para opciones."""
    primeros = {}
    for k, v in rangos.items():
        primeros[k] = v[0] if isinstance(v, (list, tuple)) and not isinstance(v[0], (int, float)) \
            else (v[0] if isinstance(v, (list, tuple)) else v)
    if not _hay_widgets():
        fn(**primeros)
        print("(sin ipywidgets: figura estática)")
        return None
    import ipywidgets as w
    from IPython.display import display
    control = w.interactive(fn, **rangos)
    display(control)
    control.update()
    return control


# --------------------------------------------------------------------------
# Esferas de Bloch interactivas (plotly): se rotan con el mouse y sobreviven
# al guardado del notebook, sin kernel corriendo
# --------------------------------------------------------------------------

def bloch_plotly(vectores, titulo="", tamano=420, mostrar_ejes=True):
    """Esfera de Bloch rotable.

    vectores: {"etiqueta": (x, y, z)} o un solo (x, y, z).
    Cada vector se dibuja con su flecha; los que tienen norma < 1 (estados
    mezcla) quedan cortos a propósito: esa es la información.
    """
    import plotly.graph_objects as go

    if not isinstance(vectores, dict):
        vectores = {"": tuple(vectores)}

    u, v = np.mgrid[0:2 * np.pi:36j, 0:np.pi:24j]
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=np.cos(u) * np.sin(v), y=np.sin(u) * np.sin(v), z=np.cos(v),
        opacity=0.13, showscale=False, hoverinfo="skip",
        colorscale=[[0, "#8fa6c4"], [1, "#8fa6c4"]]))

    t = np.linspace(0, 2 * np.pi, 96)
    ceros = np.zeros_like(t)
    for xs, ys, zs in [(np.cos(t), np.sin(t), ceros),
                       (np.cos(t), ceros, np.sin(t)),
                       (ceros, np.cos(t), np.sin(t))]:
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", hoverinfo="skip",
                                   line=dict(color="#b9c4d4", width=1.5),
                                   showlegend=False))

    if mostrar_ejes:
        for (p, q), (etiqueta_mas, etiqueta_menos) in zip(
                [((1.05, 0, 0), (-1.05, 0, 0)), ((0, 1.05, 0), (0, -1.05, 0)),
                 ((0, 0, 1.05), (0, 0, -1.05))],
                [("|+⟩", "|−⟩"), ("|+i⟩", "|−i⟩"), ("|0⟩", "|1⟩")]):
            fig.add_trace(go.Scatter3d(
                x=[p[0], q[0]], y=[p[1], q[1]], z=[p[2], q[2]], mode="lines",
                line=dict(color="#7a879b", width=2), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter3d(
                x=[p[0] * 1.13, q[0] * 1.13], y=[p[1] * 1.13, q[1] * 1.13],
                z=[p[2] * 1.13, q[2] * 1.13], mode="text",
                text=[etiqueta_mas, etiqueta_menos], textfont=dict(size=13),
                hoverinfo="skip", showlegend=False))

    colores = ["#2E5EAA", "#D1495B", "#0F7173", "#EDAE49", "#8367C7", "#66A182"]
    for i, (etiqueta, r) in enumerate(vectores.items()):
        x, y, z = float(r[0]), float(r[1]), float(r[2])
        norma = np.sqrt(x * x + y * y + z * z)
        color = colores[i % len(colores)]
        nombre = f"{etiqueta}  |r|={norma:.3f}" if etiqueta else f"|r|={norma:.3f}"
        fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z], mode="lines",
            line=dict(color=color, width=7), name=nombre,
            hovertemplate=f"{etiqueta}<br>r=({x:.3f}, {y:.3f}, {z:.3f})"
                          f"<br>|r|={norma:.3f}<extra></extra>"))
        if norma > 1e-6:
            fig.add_trace(go.Cone(
                x=[x], y=[y], z=[z], u=[x / norma * 0.18], v=[y / norma * 0.18],
                w=[z / norma * 0.18], showscale=False, hoverinfo="skip",
                colorscale=[[0, color], [1, color]], sizemode="absolute", sizeref=0.18))
        else:
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0], mode="markers", name=nombre,
                marker=dict(size=7, color=color), showlegend=False,
                hovertemplate=f"{etiqueta}<br>mezcla máxima: |r|=0<extra></extra>"))

    sin_eje = dict(showbackground=False, showgrid=False, zeroline=False,
                   showticklabels=False, title="", range=[-1.3, 1.3])
    fig.update_layout(
        title=dict(text=titulo, x=0.5, font=dict(size=14)),
        width=tamano, height=tamano, margin=dict(l=0, r=0, t=38, b=0),
        scene=dict(xaxis=sin_eje, yaxis=sin_eje, zaxis=sin_eje,
                   aspectmode="cube",
                   camera=dict(eye=dict(x=1.35, y=1.35, z=0.85))),
        showlegend=len(vectores) > 1,
        legend=dict(x=0, y=0.98, font=dict(size=11)))
    return fig


def bloch_de_estado(estado, etiqueta=""):
    """Vector de Bloch de un Statevector, DensityMatrix o QuantumCircuit de 1 qubit."""
    from qiskit.quantum_info import DensityMatrix, Pauli
    rho = DensityMatrix(estado)
    return np.array([np.real(np.trace(rho.data @ Pauli(p).to_matrix())) for p in "XYZ"])


# --------------------------------------------------------------------------
# Discos de qubit — el visor de estado del IBM Composer, que Qiskit no trae
# --------------------------------------------------------------------------

def discos_qubits(estado, titulo="", ax=None, etiquetas=None, ancho=1.45):
    """Un disco por qubit, con la codificación del IBM Quantum Composer.

    De cada qubit se muestra su estado REDUCIDO (traza parcial sobre el resto):

      · relleno azul   : P(1), llenando el disco desde abajo.
      · aguja          : la fase cuántica arg(rho_10), medida desde la derecha y
                         girando en sentido antihorario (0 a la derecha, pi/2
                         arriba, pi a la izquierda). Va de punta a punta del
                         disco gris. Si no hay coherencia la fase no está
                         definida y la aguja no se dibuja.
      · anillo negro   : la PUREZA REDUCIDA Tr[rho_j^2], que para un qubit vive
                         en [0.5, 1]. El RADIO del anillo la codifica: anillo
                         grande = puro y sin entrelazar; anillo chico = mezcla
                         máxima, o sea máximamente entrelazado con el resto.

    Esa tercera cantidad es la que hace visible el entrelazamiento: en un par de
    Bell el estado global es puro, pero cada disco tiene el anillo en el mínimo.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge
    from qiskit.quantum_info import DensityMatrix, partial_trace

    rho_total = DensityMatrix(estado)
    n = int(np.log2(rho_total.dim))
    if ax is None:
        _, ax = plt.subplots(figsize=(ancho * n + 0.6, 2.1))

    R = 0.5                                   # radio del disco gris
    for q in range(n):
        rho = partial_trace(rho_total, [k for k in range(n) if k != q]).data
        p1 = float(np.real(rho[1, 1]))
        coh = rho[1, 0]                       # rho[1,0]: su fase ES la fase relativa
        pureza = float(np.clip(np.real(np.trace(rho @ rho)), 0.5, 1.0))
        cx = q * ancho

        ax.add_patch(Circle((cx, 0), R, facecolor="0.93", edgecolor="none"))

        if p1 > 1e-9:                         # relleno azul = P(1), desde abajo
            theta = np.degrees(np.arcsin(np.clip(2 * p1 - 1, -1, 1)))
            ax.add_patch(Wedge((cx, 0), R, 180 + theta, 360 - theta,
                               facecolor=COLOR_IDEAL, alpha=0.85, lw=0))

        if abs(coh) > 1e-6:                   # aguja de fase, de largo completo
            fase = np.angle(coh)
            ax.plot([cx, cx + R * np.cos(fase)], [0, R * np.sin(fase)],
                    color="0.15", lw=2.0, solid_capstyle="round", zorder=5)
        ax.plot([cx], [0], marker="o", ms=3, color="0.15", zorder=6)

        # anillo negro: su RADIO es la pureza reducida, de [0.5,1] a [0.45R, 1.1R]
        r_anillo = R * (0.45 + 0.65 * (pureza - 0.5) / 0.5)
        ax.add_patch(Circle((cx, 0), r_anillo, fill=False, lw=2.2,
                            edgecolor="0.1", zorder=7))
        nombre = etiquetas[q] if etiquetas else f"q{q}"
        ax.text(cx, -0.92, nombre, ha="center", fontsize=9.5, fontweight="bold")
        ax.text(cx, -1.12, f"P(1) = {p1:.2f}", ha="center", fontsize=7.5, color="0.35")
        ax.text(cx, -1.30, f"pureza {pureza:.2f}", ha="center", fontsize=7.5,
                color="0.15" if pureza < 0.99 else "0.5")

    ax.set_xlim(-0.85, ancho * (n - 1) + 0.85)
    ax.set_ylim(-1.48, 0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    if titulo:
        ax.set_title(titulo, fontsize=10.5, fontweight="bold")
    return ax


# --------------------------------------------------------------------------
# Recorrer el circuito barrera por barrera
# --------------------------------------------------------------------------

def estados_por_etapa(qc, shots: int = 4096, incluir_final: bool = True) -> dict:
    """Estado del sistema en cada barrera numerada del circuito.

    Devuelve {etiqueta_de_barrera: DensityMatrix}. Usa matriz densidad y no
    vector de estado a propósito: así funciona también con mediciones a mitad
    de circuito, donde el estado deja de ser puro.

    El promedio sobre shots es lo que convierte una medición en su descripción
    de ensamble: después de medir en Z, el estado promedio es la MEZCLA
    diag(p0, p1), no la rama que salió en un tiro particular. Esa es la
    descripción correcta de "el aparato midió pero no miramos el resultado".
    """
    from qiskit.quantum_info import DensityMatrix
    from qiskit_aer import AerSimulator
    import qiskit_aer.library  # noqa: F401  (agrega save_density_matrix)

    if qc.num_qubits > 8:
        raise ValueError(f"{qc.num_qubits} qubits: la matriz densidad son "
                         f"{4 ** qc.num_qubits} elementos. Usá un circuito más chico.")

    nuevo = qc.copy_empty_like()
    etiquetas, vistas = [], set()
    for inst in qc.data:
        nuevo.append(inst.operation, inst.qubits, inst.clbits)
        if inst.operation.name == "barrier":
            lab = inst.operation.label or f"etapa {len(etiquetas) + 1}"
            while lab in vistas:
                lab += "'"
            vistas.add(lab)
            etiquetas.append(lab)
            nuevo.save_density_matrix(label=lab)
    if incluir_final and (not etiquetas or nuevo.data[-1].operation.name != "save_density_matrix"):
        lab = "final"
        etiquetas.append(lab)
        nuevo.save_density_matrix(label=lab)

    datos = AerSimulator(method="density_matrix").run(nuevo, shots=shots).result().data(0)
    return {lab: DensityMatrix(np.asarray(datos[lab])) for lab in etiquetas}


def explorador_de_etapas(qc, shots: int = 4096, titulo: str = "",
                         etiquetas_qubits=None, texto: dict = None):
    """Deslizador que recorre las barreras del circuito mostrando el estado.

    En cada etapa dibuja los discos por qubit (relleno = P(1), aguja = fase,
    borde = pureza). `texto` permite poner una línea de explicación por etapa:
    {"①": "salida del horno: rho = I/2", ...}
    """
    import matplotlib.pyplot as plt

    etapas = estados_por_etapa(qc, shots=shots)
    nombres = list(etapas)

    def dibujar(etapa):
        rho = etapas[etapa]
        n = int(np.log2(rho.dim))
        fig, ax = plt.subplots(figsize=(1.35 * n + 1.2, 2.3))
        pureza_global = float(np.real(np.trace(rho.data @ rho.data)))
        discos_qubits(rho, ax=ax, etiquetas=etiquetas_qubits,
                      titulo=f"{titulo}  —  etapa {etapa}"
                             f"   (pureza global {pureza_global:.3f})")
        plt.show()
        if texto and etapa in texto:
            print(texto[etapa])

    dibujar(nombres[0])
    if not _hay_widgets():
        print("(sin ipywidgets: se muestra solo la primera etapa)")
        return None
    import ipywidgets as w
    from IPython.display import display

    sel = w.SelectionSlider(options=nombres, value=nombres[0], description="etapa",
                            continuous_update=False, layout=w.Layout(width="520px"))
    salida = w.interactive_output(dibujar, {"etapa": sel})
    display(sel, salida)
    return salida


_NOISE_CACHE = {}


def correr_ruidoso_rapido(circuitos, backend_falso=None, shots=4096):
    """Como correr_ruidoso, pero sin mapear al chip completo.

    Transpila solo a la BASE DE COMPUERTAS del modelo de ruido, sin coupling
    map. Para circuitos de uno o dos qubits el resultado es el mismo —no hacen
    falta SWAPs— y evita transpilar contra un backend de 156 qubits, que para
    miles de circuitos chicos es el cuello de botella (llega a ser 100x más
    lento).

    Lo que NO captura: el costo de ruteo. Para circuitos anchos usá
    correr_ruidoso.
    """
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime.fake_provider import FakeFez

    if backend_falso is None:
        backend_falso = FakeFez()
    clave = backend_falso.name
    if clave not in _NOISE_CACHE:
        nm = NoiseModel.from_backend(backend_falso)
        _NOISE_CACHE[clave] = (nm, AerSimulator(noise_model=nm))
    nm, sim = _NOISE_CACHE[clave]

    uno = not isinstance(circuitos, (list, tuple))
    lista = [circuitos] if uno else list(circuitos)
    tq = transpile(lista, basis_gates=nm.basis_gates, optimization_level=1)
    res = sim.run(tq, shots=shots).result()
    salida = [res.get_counts(i) for i in range(len(lista))]
    return salida[0] if uno else salida


# --------------------------------------------------------------------------
# Cronograma real del chip (scheduler_timing)
# --------------------------------------------------------------------------

def parsear_timing(timing) -> list[dict]:
    """Convierte el scheduler_timing crudo de IBM en filas manejables.

    El dato viene como lista de strings separados por comas:
        Branch, Instruction, Channel, T0, Duration
    con T0 y Duration en unidades de dt (1 dt = 4 ns en Heron).

    'Branch' dice si la instrucción está en el camino principal o dentro de una
    rama condicional (then/else) — y ese campo es el que permite medir cuánto
    tarda el chip en decidir.
    """
    filas = []
    for linea in timing:
        partes = [p.strip() for p in str(linea).split(",")]
        if len(partes) < 5:
            continue
        rama, instr, canal = partes[0], partes[1], partes[2]
        try:
            t0, dur = int(float(partes[3])), int(float(partes[4]))
        except ValueError:
            continue
        filas.append({"rama": rama, "instruccion": instr, "canal": canal,
                      "t0": t0, "duracion": dur, "fin": t0 + dur})
    return sorted(filas, key=lambda f: f["t0"])


def resumen_timing(filas, dt: float = 4e-9) -> dict:
    """Las tres cifras que importan, en segundos.

    - duracion_total   : cuánto duró el circuito de punta a punta.
    - medicion_tipica  : cuánto tarda una medición (el costo del feed-forward).
    - latencia_decision: el hueco entre que termina la última medición del camino
      principal y arranca la primera instrucción condicionada. Es el tiempo que
      el chip pasa decidiendo, y el qubit decohera durante todo ese rato.
    """
    if not filas:
        return {}
    total = max(f["fin"] for f in filas)
    mediciones = [f for f in filas if "measure" in f["instruccion"].lower()]
    # OJO: el orden de los campos está documentado, pero las ETIQUETAS de rama
    # ('main', 'then', ...) son una suposición hasta ver un job real. Por eso
    # marcamos como condicional lo que parece condicional, y además devolvemos
    # las etiquetas encontradas para poder revisarlas a mano.
    etiquetas = sorted({f["rama"] for f in filas})
    def es_condicional(r):
        r = r.lower()
        if r in ("main", "principal", "", "-", "none"):
            return False
        return any(k in r for k in ("then", "else", "if", "case", "body",
                                    "branch", "true", "false")) or True
    condicionadas = [f for f in filas if es_condicional(f["rama"])]
    res = {
        "duracion_total_dt": total,
        "duracion_total_s": total * dt,
        "n_instrucciones": len(filas),
        "n_condicionadas": len(condicionadas),
        "canales": sorted({f["canal"] for f in filas}),
        "etiquetas_de_rama": etiquetas,
    }
    if mediciones:
        res["medicion_dt"] = int(np.median([f["duracion"] for f in mediciones]))
        res["medicion_s"] = res["medicion_dt"] * dt
    if condicionadas and mediciones:
        inicio_cond = min(f["t0"] for f in condicionadas)
        fin_medicion = max((f["fin"] for f in mediciones
                            if f["fin"] <= inicio_cond), default=None)
        if fin_medicion is not None:
            res["latencia_decision_dt"] = inicio_cond - fin_medicion
            res["latencia_decision_s"] = (inicio_cond - fin_medicion) * dt
    return res


def dibujar_timing(filas, dt: float = 4e-9, ax=None, titulo=""):
    """Diagrama de Gantt del cronograma: una fila por canal del chip.

    Los canales 'AWGR<instrumento>_<qubit>' son de lectura, y los qubits que
    comparten instrumento comparten feedline: por eso no se puede medir a
    cualquiera en paralelo sin que se afecten.
    """
    import matplotlib.pyplot as plt

    if not filas:
        print("no hay datos de timing")
        return None
    canales = sorted({f["canal"] for f in filas})
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 0.45 * len(canales) + 1.6))
    colores = {"measure": COLOR_RUIDO, "reset": PALETA[3], "barrier": "0.85"}
    for f in filas:
        y = canales.index(f["canal"])
        nombre = f["instruccion"].split()[0].lower()
        color = next((c for k, c in colores.items() if k in nombre), COLOR_IDEAL)
        condicional = f["rama"].lower() not in ("main", "principal", "")
        ax.barh(y, max(f["duracion"], 1) * dt * 1e9, left=f["t0"] * dt * 1e9,
                height=0.62, color=color, alpha=0.95,
                edgecolor=COLOR_HARD if condicional else "none",
                linewidth=1.8 if condicional else 0)
    ax.set_yticks(range(len(canales)))
    ax.set_yticklabels(canales, fontsize=8, family="monospace")
    ax.set_xlabel("tiempo [ns]")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    r = resumen_timing(filas, dt)
    if len(r.get("etiquetas_de_rama", [])) > 1:
        print("etiquetas de rama encontradas:", r["etiquetas_de_rama"],
              "\n  (si alguna que NO sea condicional quedó marcada con borde, "
              "revisá es_condicional en utils.resumen_timing)")
    ax.set_title(titulo or f"Cronograma real · {r.get('duracion_total_s', 0) * 1e6:.2f} µs "
                           f"· {r['n_instrucciones']} instrucciones "
                           f"({r['n_condicionadas']} condicionadas, con borde)",
                 fontsize=10.5, fontweight="bold")
    return ax


def pedir_timing(sampler_options) -> None:
    """Activa el flag experimental que hace que IBM devuelva el cronograma.

    Es API experimental: puede cambiar sin aviso. Y no tiene equivalente en
    simulador — el cronograma lo arma el compilador de IBM, así que este dato
    solo existe si mandás el job de verdad.
    """
    sampler_options.experimental = {"execution": {"scheduler_timing": True}}


def timing_de_resultado(resultado, i: int = 0):
    """Saca el cronograma del metadata de un PrimitiveResult."""
    try:
        comp = resultado[i].metadata["compilation"]["scheduler_timing"]
    except (KeyError, TypeError):
        print("este job no trae scheduler_timing: ¿le pusiste el flag al mandarlo?")
        return None, None
    return parsear_timing(comp["timing"]), comp
