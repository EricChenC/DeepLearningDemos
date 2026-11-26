




print("Hello Deep Learning")

# ==========================================
# Simple Neural Network Training
# Using Mean Squared Error (MSE)
# ==========================================

# 1. Initialize
W = 10
x = 1
target = 5

learning_rate = 0.1
iteration = 0

# 2. Training loop
while True:

    iteration += 1

    # --------------------------------------
    # FORWARD
    # --------------------------------------

    q = W * x

    # Residual
    L = target - q

    # Loss function
    J = L ** 2

    print(f"\nIteration: {iteration}")
    print(f"Forward:")
    print(f"  W = {W:.8f}")
    print(f"  q = W * x = {q:.8f}")
    print(f"  L = target - q = {L:.8f}")
    print(f"  J = L^2 = {J:.8f}")

    # --------------------------------------
    # STOP CONDITION
    # --------------------------------------

    if abs(L) < 1e-8:
        print("\nTraining completed!")
        break

    # --------------------------------------
    # BACKWARD
    # --------------------------------------

    dJ_dL = 2 * L
    dL_dq = -1
    dq_dW = x

    # Chain rule
    dJ_dW = dJ_dL * dL_dq * dq_dW

    print(f"Backward:")
    print(f"  dJ/dL = {dJ_dL:.8f}")
    print(f"  dL/dq = {dL_dq}")
    print(f"  dq/dW = {dq_dW}")
    print(f"  dJ/dW = {dJ_dW:.8f}")

    # --------------------------------------
    # UPDATE
    # --------------------------------------

    W = W - learning_rate * dJ_dW

    print(f"Update:")
    print(f"  New W = {W:.8f}")

    if iteration >= 1000:
        print("\nMaximum iterations reached!")
        break


print("Hello Deep Learning")
