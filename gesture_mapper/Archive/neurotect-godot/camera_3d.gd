extends Camera3D

@export var speed: float = 10.0
@export var mouse_sensitivity: float = 0.2  # Adjust for sensitivity

var rotation_x: float = 0.0  # Pitch (up/down rotation)
var rotation_y: float = 0.0  # Yaw (left/right rotation)

func _ready():
    Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)  # Lock the cursor

func _unhandled_input(event):
    if event is InputEventMouseMotion:
        rotate_camera(event.relative)

func _physics_process(delta: float) -> void:
    var velocity = Vector3.ZERO

    # UP and DOWN for up/down (Y axis)
    if Input.is_key_pressed(KEY_UP):
        velocity.y += 1.0
    if Input.is_key_pressed(KEY_DOWN):
        velocity.y -= 1.0

    # Left and Right arrows for left/right movement (X axis)
    if Input.is_key_pressed(KEY_LEFT) or Input.is_key_pressed(KEY_A):
        velocity.x -= 1.0
    if Input.is_key_pressed(KEY_RIGHT) or Input.is_key_pressed(KEY_D):
        velocity.x += 1.0

    # W and S arrows for forward/backward movement (Z axis)
    if Input.is_key_pressed(KEY_W):
        velocity.z -= 1.0  # Move forward
    if Input.is_key_pressed(KEY_S):
        velocity.z += 1.0  # Move backward

    if velocity != Vector3.ZERO:
        velocity = velocity.normalized() * speed * delta
        translate(velocity)

func rotate_camera(mouse_motion: Vector2):
    rotation_y -= deg_to_rad(mouse_motion.x * mouse_sensitivity)  # Yaw (left/right)
    rotation_x -= deg_to_rad(mouse_motion.y * mouse_sensitivity)  # Pitch (up/down)

    # Clamp the pitch so you can't look too far up/down
    rotation_x = clamp(rotation_x, deg_to_rad(-89), deg_to_rad(89))

    # Apply rotation
    rotation_degrees.x = rad_to_deg(rotation_x)
    rotation_degrees.y = rad_to_deg(rotation_y)
