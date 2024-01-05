{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "def zscore(x):\n",
    "    mean = np.mean(x, axis=0)\n",
    "    sigma = np.std(x, axis=0)\n",
    "    x_norm = (x - mean) / sigma\n",
    "    return x_norm\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict(x, w, b):\n",
    "    y_pred = x @ w + b\n",
    "    return y_pred\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def compute_cost(x, y, y_pred):\n",
    "    m, n = x.shape\n",
    "    err = y_pred - y\n",
    "    cost = err * err\n",
    "    cost = np.sum(cost)\n",
    "    return cost / (2 * m)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def gradient(x, y, y_pred):\n",
    "    err = y_pred - y\n",
    "    dw = x.T @ err\n",
    "    db = np.sum(err)\n",
    "    return dw / m, db / m\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def gradient_descent(x, y, w, b, lr, itera):\n",
    "    costs = []  # List to store the cost for each iteration\n",
    "    for i in range(itera + 1):\n",
    "        y_pred = predict(x, w, b)\n",
    "        dw, db = gradient(x, y, y_pred)\n",
    "        w = w - lr * dw\n",
    "        b = b - lr * db\n",
    "        if i % 1000 == 0:\n",
    "            cost = compute_cost(x, y, y_pred)\n",
    "            costs.append(cost)\n",
    "            print(f\"Cost after {i} iterations: {cost}\")\n",
    "    return w, b, costs\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_cost_convergence(costs):\n",
    "    iterations = list(range(0, len(costs) * 1000, 1000))\n",
    "    plt.plot(iterations, costs, marker='o')\n",
    "    plt.title(\"Cost Convergence during Gradient Descent\")\n",
    "    plt.xlabel(\"Iterations\")\n",
    "    plt.ylabel(\"Cost\")\n",
    "    plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "def main():\n",
    "    # Load data\n",
    "    data = pd.read_csv(\"D:\\\\HP\\\\users\\\\OneDrive\\\\Desktop\\\\AI ML\\\\train and test\\\\linear_train.csv\")\n",
    "    \n",
    "    # Display the size of the data\n",
    "    print(f\"Data size: {data.shape}\")\n",
    "\n",
    "    # Extract features and labels\n",
    "    x = data.iloc[:45000, 1:21].values\n",
    "    y = data.iloc[:45000, 21:22].values\n",
    "\n",
    "    # Initialize weights and bias\n",
    "    m, n = x.shape\n",
    "    w = np.zeros((n, 1))\n",
    "    b = 0\n",
    "\n",
    "    # Set hyperparameters\n",
    "    lr = 8e-4\n",
    "    itera = 12000\n",
    "\n",
    "    # Perform gradient descent\n",
    "    w, b, costs = gradient_descent(zscore(x), y, w, b, lr, itera)\n",
    "\n",
    "    # Make predictions on test data\n",
    "    x_test = data.iloc[49990:, 1:21].values\n",
    "    y_pred = predict(zscore(x_test), w, b)\n",
    "\n",
    "    print(\"Predictions for the test data:\")\n",
    "    print(y_pred)\n",
    "\n",
    "    # Plot cost convergence\n",
    "    plot_cost_convergence(costs)\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
