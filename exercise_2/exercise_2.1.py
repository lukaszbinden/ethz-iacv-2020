
camera_width = 640
camera_height = 480

film_back_width = 1.417
film_back_height = 0.945

x_center = 320
y_center = 240

P_1 = (-0.023, -0.261, 2.376)
p_11 = P_1[0]
p_12 = P_1[1]
p_13 = P_1[2]

P_2 = (0.659, -0.071, 2.082)
p_21 = P_2[0]
p_22 = P_2[1]
p_23 = P_2[2]

p_1_prime = (52, 163)
x_1 = p_1_prime[0]
y_1 = p_1_prime[1]

p_2_prime = (218, 216)
x_2 = p_2_prime[0]
y_2 = p_2_prime[1]

f = 1.378
k_x = camera_width / film_back_width
k_y = camera_height / film_back_height

# f_k_x = f * k_x
f_k_x = f
# f_k_y = f * k_y
f_k_y = f

u_1_prime = (x_1 - x_center) / k_x
v_1_prime = (y_1 - y_center) / k_y

u_2_prime = (x_2 - x_center) / k_x
v_2_prime = (y_2 - y_center) / k_y

c_1_prime = (f_k_x * p_21 + (p_13 - p_23) * u_2_prime - u_2_prime/u_1_prime * f_k_x * p_11) / (f_k_x * (1 - u_2_prime/u_1_prime))

c_2_prime = (f_k_y * p_22 - (p_23 - (p_13*u_1_prime - f_k_x*(p_11 - c_1_prime))/u_1_prime) * v_2_prime) / f_k_y

c_2_prime_alt = (f_k_y * p_12 - (p_13 - (p_13*u_1_prime - f_k_x*(p_11 - c_1_prime))/u_1_prime) * v_1_prime) / f_k_y

c_3_prime = p_13 - (f_k_x / u_1_prime) * (p_11 - c_1_prime)

rho_1_prime = p_13 - c_3_prime
rho_2_prime = p_23 - c_3_prime

print(f"C' = ({c_1_prime}, {c_2_prime}, {c_3_prime})")
print(f"c_2_prime_alt = {c_2_prime_alt}")

print(f"rho_1_prime = {rho_1_prime}")
print(f"rho_2_prime = {rho_2_prime}")

print("------------------")
r_11 = f_k_x * (p_11 - c_1_prime)
r_12 = f_k_y * (p_12 - c_2_prime)
r_13 = 1 * (p_13 - c_3_prime)

l_11 = rho_1_prime * u_1_prime
l_12 = rho_1_prime * v_1_prime
l_13 = rho_1_prime * 1

print(f"L: ({l_11}, {l_12}, {l_13})")
print(f"R: ({r_11}, {r_12}, {r_13})")

print("------------------")
r_21 = f_k_x * (p_21 - c_1_prime)
r_22 = f_k_y * (p_22 - c_2_prime)
r_23 = 1 * (p_23 - c_3_prime)

l_21 = rho_2_prime * u_2_prime
l_22 = rho_2_prime * v_2_prime
l_23 = rho_2_prime * 1

print(f"L: ({l_11}, {l_12}, {l_13})")
print(f"R: ({r_11}, {r_12}, {r_13})")