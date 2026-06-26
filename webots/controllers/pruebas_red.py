import os, numpy as np, torch, torch.nn as nn, joblib, csv
from controller import Supervisor

class MantisPredictiveRNN(nn.Module):
    def __init__(self, input_dim=37, hidden_dim=128, goal_dim=19):
        super(MantisPredictiveRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.task_dense = nn.Sequential(nn.Linear(hidden_dim + goal_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.final_dense = nn.Linear(hidden_dim, input_dim * 2)

    def forward(self, x_task, x_babble, goal_vector):
        x_all = torch.cat([x_task, x_babble], dim=0)
        lstm_out, _ = self.lstm(x_all)
        out_task, out_babble = torch.split(lstm_out, [x_task.size(0), x_babble.size(0)], dim=0)
        task_merged = torch.cat([out_task, goal_vector], dim=-1)
        task_features = self.task_dense(task_merged)
        features_all = torch.cat([task_features, out_babble], dim=0)
        final_out = self.final_dense(features_all)
        return torch.chunk(final_out, chunks=2, dim=-1)

# Inicialización
robot = Supervisor()
time_step = int(robot.getBasicTimeStep())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preparar carpeta de réplicas
#carpeta_replicas = "replicas_inner"
#carpeta_replicas = "replicas_outer"
carpeta_replicas = "replicas_moving"
if not os.path.exists(carpeta_replicas):
    os.makedirs(carpeta_replicas)

# Carga de modelos
scaler = joblib.load('traductor_escalas_mantis.pkl')
modelo = MantisPredictiveRNN().to(DEVICE)
modelo.load_state_dict(torch.load("mantis_lfd_mejor_modelo.pth", map_location=DEVICE))
modelo.eval()

# Dispositivos
receiver = robot.getDevice("receiver")
receiver.enable(time_step)
motors = [robot.getDevice(name) for name in ["RPC", "RPF", "RPT", "RMC", "RMF", "RMT", "RAC", "RAF", "RAT", "LPC", "LPF", "LPT", "LMC", "LMF", "LMT", "LAC", "LAF", "LAT"]]
touch_sensors = [robot.getDevice(name) for name in ["centro", "zona_frente_1_der", "zona_frente_1_izq", "zona_atras_1_der", "zona_atras_1_izq", "zona_izq_1", "zona_der_1", "zona_izq_2_frente", "zona_izq_2_atras", "zona_der_2_frente", "zona_der_2_atras", "zona_frente_2_der", "zona_frente_2_izq", "zona_atras_2_der", "zona_atras_2_izq", "zona_frente_3_der", "zona_frente_3_izq", "zona_atras_3_der", "zona_atras_3_izq"]]
for s in touch_sensors: s.enable(time_step)

pelota_node = robot.getFromDef("PELOTA")
mantis_node = robot.getFromDef("MANTIS")

telemetria = []
contador_episodios = 1
pasos_episodio = 0
historial_estados = []
WINDOW_SIZE = 20 #8 #4
base_goal = [1.0] + [0.0] * 18
goal_tensor = torch.tensor(base_goal, dtype=torch.float32).view(1, 1, 19).repeat(1, WINDOW_SIZE, 1).to(DEVICE)
x_babble_dummy = torch.empty(0, WINDOW_SIZE, 37, dtype=torch.float32).to(DEVICE)

# Bucle principal
while robot.step(time_step) != -1:
    pasos_episodio += 1
    hubo_reset = False
    while receiver.getQueueLength() > 0:
        if receiver.getString() == "RESET": hubo_reset = True
        receiver.nextPacket()
        
    if hubo_reset:
        duracion = pasos_episodio * (time_step / 1000.0)
        pos_p = pelota_node.getPosition()
        if duracion >= 59.5:
            nombre = os.path.join(carpeta_replicas, f"ep{contador_episodios}_dur{duracion:.1f}s_PX{pos_p[0]:.3f}_PY{pos_p[1]:.3f}.csv")
            encabezado = [
                "Tiempo_Webots", "Pelota_X", "Pelota_Y", "Pelota_Z", "Mantis_X", "Mantis_Y", "Mantis_Z", 
                "Mantis_Rot_0", "Mantis_Rot_1", "Mantis_Rot_2", "Mantis_Rot_3",
                "Pos_Motor_RPC", "Pos_Motor_RPF", "Pos_Motor_RPT", "Pos_Motor_RMC", "Pos_Motor_RMF", "Pos_Motor_RMT", 
                "Pos_Motor_RAC", "Pos_Motor_RAF", "Pos_Motor_RAT", "Pos_Motor_LPC", "Pos_Motor_LPF", "Pos_Motor_LPT", 
                "Pos_Motor_LMC", "Pos_Motor_LMF", "Pos_Motor_LMT", "Pos_Motor_LAC", "Pos_Motor_LAF", "Pos_Motor_LAT",
                "Sensor_centro", "Sensor_zona_frente_1_der", "Sensor_zona_frente_1_izq", "Sensor_zona_atras_1_der", 
                "Sensor_zona_atras_1_izq", "Sensor_zona_izq_1", "Sensor_zona_der_1", "Sensor_zona_izq_2_frente", 
                "Sensor_zona_izq_2_atras", "Sensor_zona_der_2_frente", "Sensor_zona_der_2_atras", "Sensor_zona_frente_2_der", 
                "Sensor_zona_frente_2_izq", "Sensor_zona_atras_2_der", "Sensor_zona_atras_2_izq", "Sensor_zona_frente_3_der", 
                "Sensor_zona_frente_3_izq", "Sensor_zona_atras_3_der", "Sensor_zona_atras_3_izq"
            ]
            with open(nombre, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(encabezado)
                writer.writerows(telemetria)
            print(f"*** Guardado limpio en {carpeta_replicas}: {os.path.basename(nombre)}")
        telemetria, pasos_episodio, historial_estados = [], 0, []
        contador_episodios += 1
        continue

    # Inferencia
    lista_estado = [m.getTargetPosition() for m in motors] + [1.0 if s.getValue() > 0 else 0.0 for s in touch_sensors]
    obs_scaled = scaler.transform(np.array(lista_estado, dtype=np.float32).reshape(1, -1)).astype(np.float32).flatten()
    historial_estados.append(obs_scaled)
    if len(historial_estados) > WINDOW_SIZE: historial_estados.pop(0)
    while len(historial_estados) < WINDOW_SIZE: historial_estados.insert(0, obs_scaled)
    
    with torch.no_grad():
        mu, _ = modelo(torch.tensor(np.array(historial_estados), dtype=torch.float32).unsqueeze(0).to(DEVICE), x_babble_dummy, goal_tensor)
        pred = scaler.inverse_transform(mu[0, -1, :].cpu().numpy().reshape(1, -1)).flatten()
    for i in range(18): 
        motors[i].setPosition(pred[i])
    
    # Registro (48 campos limpios)
    p = pelota_node.getPosition()
    m_pos = mantis_node.getPosition()
    m_rot = mantis_node.getOrientation()
    telemetria.append([robot.getTime()] + p + m_pos + m_rot[:4] + [m.getTargetPosition() for m in motors] + [1.0 if s.getValue() > 0 else 0.0 for s in touch_sensors])