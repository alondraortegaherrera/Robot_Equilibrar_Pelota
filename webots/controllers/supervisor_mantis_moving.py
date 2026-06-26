import math
import random
import csv
from controller import Supervisor

# ==========================================
# 1. CONFIGURACIÓN DEL LABORATORIO DE PRUEBAS
# ==========================================
EPISODIOS_TOTALES_PRUEBA = 5_000
TIEMPO_MAXIMO_EPISODIO = 60.0
NOMBRE_ARCHIVO_LOG = "resultados_moving.csv"
TIEMPO_EMPUJON = 4.0
FUERZA_IMPULSO = 35.0

# Inicializar objeto Supervisor
supervisor = Supervisor()
time_step = int(supervisor.getBasicTimeStep())
emitter = supervisor.getDevice("emitter")
receiver = supervisor.getDevice("receiver")
receiver.enable(time_step)

robot_node = supervisor.getFromDef("MANTIS")
pelota_node = supervisor.getFromDef("PELOTA")
pelota_translation_field = pelota_node.getField("translation")

X_MIN, X_MAX = -9.09, -7.67
Y_MIN, Y_MAX = -0.28, 0.28
ALTURA_Z, Z_LIMITE_CAIDA = 1.1, 0.82

# Variables de control
contador_episodios = 1
historial_datos_crudos = []
empujon_aplicado = False
dir_registrada = "N/A"

# Posición inicial para el primer episodio
x_aleatoria = random.uniform(X_MIN, X_MAX)
y_aleatoria = random.uniform(Y_MIN, Y_MAX)
pelota_translation_field.setSFVec3f([x_aleatoria, y_aleatoria, ALTURA_Z])
emitter.send(f"PELOTA:{x_aleatoria:.5f},{y_aleatoria:.5f}".encode('utf-8'))

tiempo_inicio_episodio = supervisor.getTime()

print(f"\n=============================================================")
print(f" [SUPERVISOR] Iniciando validación escenario MOVING ({EPISODIOS_TOTALES_PRUEBA} eps)")
print(f"===============================================================\n")

# Bucle principal
while supervisor.step(time_step) != -1:
    tiempo_transcurrido = supervisor.getTime() - tiempo_inicio_episodio
    while receiver.getQueueLength() > 0: receiver.nextPacket()
    
    # Lógica del empujón
    if not empujon_aplicado and tiempo_transcurrido >= TIEMPO_EMPUJON:
        pose = robot_node.getPose()
        opciones = [
            (pose[0], pose[1], "DERECHA"),
            (-pose[0], -pose[1], "IZQUIERDA"),
            (pose[4], pose[5], "ADELANTE"),
            (-pose[4], -pose[5], "ATRAS")
        ]
        dx, dy, dir_registrada = random.choice(opciones)
        pelota_node.addForce([dx * FUERZA_IMPULSO, dy * FUERZA_IMPULSO, 0], True)
        empujon_aplicado = True
    
    pos_global_pelota = pelota_node.getPosition()
    reinicio_por_caida = (pos_global_pelota[2] < Z_LIMITE_CAIDA)
    reinicio_por_tiempo = (tiempo_transcurrido >= TIEMPO_MAXIMO_EPISODIO)
    
    if reinicio_por_caida or reinicio_por_tiempo:
        if reinicio_por_caida:
            resultado_actual = "CAIDA"
            print(f"[Episodio {contador_episodios}/{EPISODIOS_TOTALES_PRUEBA}] FRACASO - Duración: {tiempo_transcurrido:.2f}s | Dir: {dir_registrada} | Z Global: {pos_global_pelota[2]:.2f}m")
        else:
            resultado_actual = "EXITO"
            print(f"[Episodio {contador_episodios}/{EPISODIOS_TOTALES_PRUEBA}] ¡ÉXITO! - Completó los {TIEMPO_MAXIMO_EPISODIO}s | Dir: {dir_registrada}")
            
        historial_datos_crudos.append((contador_episodios, resultado_actual, round(tiempo_transcurrido, 4), dir_registrada))
        
        if contador_episodios >= EPISODIOS_TOTALES_PRUEBA:
            with open(NOMBRE_ARCHIVO_LOG, 'w', newline='') as f:
                csv.writer(f).writerows([["ID", "Resultado", "Duracion", "Direccion"]] + historial_datos_crudos)
            print(f"\n[SUPERVISOR] Experimento finalizado. Archivo guardado: {NOMBRE_ARCHIVO_LOG}")
            supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
            break
            
        # Reset
        contador_episodios += 1
        empujon_aplicado = False
        dir_registrada = "N/A"
        x, y = random.uniform(X_MIN, X_MAX), random.uniform(Y_MIN, Y_MAX)
        
        supervisor.simulationReset()
        supervisor.simulationResetPhysics()
        supervisor.step(time_step) 
        
        pelota_node = supervisor.getFromDef("PELOTA")
        pelota_node.getField("translation").setSFVec3f([x, y, ALTURA_Z])
        pelota_node.resetPhysics()
        
        emitter.send("RESET".encode('utf-8'))
        emitter.send(f"PELOTA:{x:.5f},{y:.5f}".encode('utf-8'))
        
        tiempo_inicio_episodio = supervisor.getTime()