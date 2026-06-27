import math

# training dataset
# Format: [Open Ports, Latency ms], Label
dataset = [
    [[80, 2], "Server"],        # Web server (High ports, ultra-low latency)
    [[22, 4], "Server"],        # SSH server
    [[443, 3], "Server"],       # HTTPS server
    [[2, 45], "Workstation"],   # Laptop on Wi-Fi (Low ports, high latency)
    [[1, 30], "Workstation"],   # Desktop
    [[5, 60], "Workstation"]    # Mobile device
]

port_min, port_max = 1, 443
lat_min, lat_max = 2, 60

def norm(ports, lat):
	norm_ports = (ports - port_min) / (port_max - port_min)
	norm_lat = (lat - lat_min) / (lat_max - lat_min)
	return norm_ports, norm_lat

def calculate_dist(p1, p2):
	return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def classify_nn(new_ports, new_lat):
	new_point = norm(new_ports, new_lat)
	closest_dist = float("inf")
	pred_label = None
	for known_features, label in dataset:
		known_point = norm(known_features[0], known_features[1])
		dist = calculate_dist(new_point, known_point)
		
		if dist < closest_dist:
			closest_dist = dist
			pred_label = label

	return pred_label, closest_dist

test_ports = 45
test_lat = 5

prediction, dist = classify_nn(test_ports, test_lat)
print(f"Predicted Device Type: {prediction}")
print(f"Mathematical Distance: {dist:.4f}")
