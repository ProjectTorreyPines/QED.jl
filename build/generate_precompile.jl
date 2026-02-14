using QED
using JSON
filename = joinpath(dirname(dirname(abspath(@__FILE__))), "sample", "ods_small.json")
data = JSON.parsefile(filename)
QI = from_imas(data)
η = η_imas(data)

Jt_R(QI)
Ip(QI)
JB(QI)

# --- diffuse: public API (3 BC variants) ---
Q1a = diffuse(QI, η, 0.001, 10)                # default BC
Q1b = diffuse(QI, η, 0.001, 10; Vedge=0.1)     # Vedge BC
Q1c = diffuse(QI, η, 0.001, 10; Ip=1e6)        # Ip BC

# --- _diffuse: private API with precomputed T, Y (3 BC variants) ---
T = QED.define_T(QI)
Y = QED.define_Y(QI, η)
Q2a = QED._diffuse(QI, η, 0.001, 10, T, Y)              # default BC
Q2b = QED._diffuse(QI, η, 0.001, 10, T, Y; Vedge=0.1)   # Vedge BC
Q2c = QED._diffuse(QI, η, 0.001, 10, T, Y; Ip=1e6)      # Ip BC

# --- steady_state: public API (3 BC variants) ---
Q3a = steady_state(QI, η)                       # default BC
Q3b = steady_state(QI, η; Vedge=0.0)            # Vedge BC
Q3c = steady_state(QI, η; Ip=1e6)               # Ip BC

# --- _steady_state: private API with precomputed Y (3 BC variants) ---
Y = QED.define_Y(QI, η)
Q4a = QED._steady_state(QI, η, Y)               # default BC
Q4b = QED._steady_state(QI, η, Y; Vedge=0.0)    # Vedge BC
Q4c = QED._steady_state(QI, η, Y; Ip=1e6)       # Ip BC

η = η_mock()

QI = QED.QED_state(QI; JBni=x -> -1e6 * (0.9 * sin(2π * x) + 0.1))
Q5 = steady_state(QI, η; Vedge=0.0)
Y = QED.define_Y(QI, η)
Q6 = QED._steady_state(QI, η, Y; Vedge=0.0);
