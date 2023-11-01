
import math

# --- sigmoid activation function --- #
def sigmoid(x):
  return 1 / (1 + math.exp(-x))


print(sigmoid(10))
print(sigmoid(1))
print(sigmoid(-56))
print(sigmoid(0.5))




# --- tanh activation function --- #
def tanh(x):
  return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


# print(tanh(-56))
# print(tanh(50))
# print(tanh(1))




# --- relu activation function --- #
def relu(x):
    return max(0, x)


# print(relu(-100))
# print(relu(8))





# --- leaky_relu activation function --- #
def leaky_relu(x):
    return max(0.1*x, x)


# print(leaky_relu(-100))
# print(leaky_relu(8))