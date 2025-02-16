import pandas as pd
import matplotlib.pyplot as plt

def plot_training_log(file_path):
    try:
        data = pd.read_csv(file_path)
        
        

        required_columns = ['iteration', 'loss', 'best_perplexity']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in the CSV file.")

        plt.figure(figsize=(12, 6))
        plt.plot(data['iteration'], data['loss'], label='Loss', color='blue')
        plt.xlabel('Iteration')
        plt.xticks(ticks=data['iteration'][data['iteration'] % 10000 == 0])
        plt.ylabel('Loss')
        plt.yticks(ticks=[i for i in range(int(data['loss'].min()), int(data['loss'].max()) + 1, 2)])
        plt.title('Loss Over Iterations')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(data['iteration'], data['best_perplexity'], label='Best Perplexity', color='green')
        plt.xlabel('Iteration')
        plt.ylabel('Best Perplexity')
        plt.title('Best Perplexity Over Iterations')
        plt.legend()
        plt.grid(True)
        plt.show()

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    plot_training_log('training_log.csv')