using QED
using JSON
filename = joinpath(dirname(dirname(abspath(@__FILE__))), "sample", "ods_small.json")
data = JSON.parsefile(filename)
QI = from_imas(data)
η = η_imas(data)

Jt_R(QI)
Ip(QI)
JB(QI)

Q1 = diffuse(QI, η, 0.001, 10)
T = define_T(QI)
Y = define_Y(QI, η)
Q2 = diffuse(QI, η, 0.001, 10, T, Y)
Q3 = diffuse(QI, η, 0.001, 10, T, Y; Vedge=0.1)
Q4 = diffuse(QI, η, 0.001, 10, T, Y; Ip=1e6)

η = η_mock()

QI = QED_state(QI; JBni=x -> -1e6 * (0.9 * sin(2π * x) + 0.1))
Q5 = steady_state(QI, η; Vedge=0.0)
Y = define_Y(QI, η)
Q6 = steady_state(QI, η, Y; Vedge=0.0);