extends Node3D

var points: Array = []
var line_nodes: Array = []
var server: UDPServer

# List of connections between landmarks (index pairs)
var connections = [
	[0,1], [1,2], [2,3], [3,4],
	[0,5], [5,6], [6,7], [7,8],
	[0,9], [9,10], [10,11], [11,12],
	[0,13], [13,14], [14,15], [15,16],
	[0,17], [17,18], [18,19], [19,20]
]

func _ready() -> void:
	# Find and store landmark nodes.
	for i in range(21):
		var point_name = "Point" + str(i)
		var point = get_node_or_null(point_name)
		if point:
			points.append(point)
		else:
			print("⚠️ Warning: Could not find node ", point_name)
	
	# Create cylinders for each connection.
	for connection in connections:
		var cylinder = MeshInstance3D.new()
		var mesh = CylinderMesh.new()
		mesh.top_radius = 0.02
		mesh.bottom_radius = 0.02
		mesh.height = 1.0  # Will be adjusted dynamically.
		cylinder.mesh = mesh
		add_child(cylinder)
		line_nodes.append(cylinder)
	
	# Create the UDP server.
	server = UDPServer.new()
	var err = server.listen(5051)  # Change port if needed.
	if err != OK:
		print("❌ Error starting UDP server: ", err)
	else:
		print("✅ UDP server listening on port 5051")

func update_hand_positions(coords: PackedFloat32Array):
	if coords.size() != 63:
		print("❌ Error: Expected 63 coordinates, got ", coords.size())
		return
	for i in range(21):
		var x = coords[3 * i] * 70
		var y = -coords[3 * i + 1] * 70
		var z = coords[3 * i + 2] * 210
		if i < points.size():
			points[i].position = Vector3(x, y, z)
		else:
			print("❌ Error: Point index ", i, " not found")

func _process(delta: float) -> void:
	server.poll()  # Check for new packets

	# Check if there is an available connection
	while server.is_connection_available():
		var peer: PacketPeerUDP = server.take_connection()
		if peer:
			var packet: PackedByteArray = peer.get_packet()
			
			if packet.is_empty():
				print("❌ Error: Received empty packet")
				return
			
			var data: String = packet.get_string_from_utf8()
			print("📩 Data received: ", data)
			
			var str_array = data.split(", ")
			if str_array.size() != 63:
				print("⚠️ Warning: Expected 63 values, but got ", str_array.size())
			
			var float_array: PackedFloat32Array = []
			for s in str_array:
				float_array.append(s.to_float())
			
			update_hand_positions(float_array)
	
	# Update cylinder positions and orientations.
	for i in range(connections.size()):
		var index_a = connections[i][0]
		var index_b = connections[i][1]
		if index_a < points.size() and index_b < points.size():
			var pos_a = points[index_a].global_transform.origin
			var pos_b = points[index_b].global_transform.origin
			var mid_point = (pos_a + pos_b) / 2.0  
			var direction = (pos_b - pos_a).normalized()  
			var distance = pos_a.distance_to(pos_b)
			var cylinder = line_nodes[i]
			cylinder.global_transform.origin = mid_point

			var basis = Basis()
			basis.y = direction
			basis.x = Vector3.FORWARD.cross(direction).normalized()
			basis.z = basis.x.cross(basis.y).normalized()
			cylinder.global_transform.basis = basis

			cylinder.mesh.height = distance
