import os.path

from tkinter import *
import tkinter as tk
from tkinter import filedialog, Text, ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from stock_model import *



def training(frame, results_box, clicked):
    # Placeholder for analysis logic
    plot(frame, results_box, clicked)

def predict(frame, results_box, clicked):
    plot_predict(frame, results_box, clicked)

def get_predict(current, result_predict):
    out = ''
    out += 'Приведенные значения: ' + str(current) + '\n'
    out += 'Среднее прогнозируемое значение: ' + str(result_predict.mean()) + '\n'
    if current > result_predict.mean():
        out += 'Прогнозируемое среднее значение на следующие 30 дней ниже текущего значения -> Следует продавать'
    else:
        out += 'Прогнозируемое среднее значение на следующие 30 дней выше текущего значения -> Следует покупать'
    return out

def get_result(test_y, test_predict):
    out = ''
    out += 'R^2 score: ' + str(r2_score(test_y, test_predict)) + "\n"
    out += 'mean_absolute_error: ' + str(mean_absolute_error(test_y, test_predict)) + "\n"
    out += 'root mean squared error: ' + str(mean_squared_error(test_y, test_predict, squared=False)) + "\n"
    out += 'root mean squared logarithmic error: ' + str(mean_squared_log_error(test_y, test_predict, squared=False)) + "\n"
    out += 'median_absolute_error: ' + str(median_absolute_error(test_y, test_predict))
    return out
def insert_result(textbox, out):
    textbox.config(state=tk.NORMAL)
    textbox.delete("1.0", tk.END)
    textbox.insert(1.0, out)
    textbox.config(state=tk.DISABLED)

def plot(frame, results_box, clicked):
    # Clear the frame
    for widget in frame.winfo_children():
        widget.destroy()

    # get code crypto
    code = clicked.get()
    print(code)

    # Create a figure
    fig = Figure(figsize=(6, 4), dpi=100)

    # Add a subplot
    test_y, test_predict, test_loss = before_train(code)
    plot1 = fig.add_subplot(111)
    # Plot data

    plot1.plot(test_y, label='Истинные значения')
    plot1.plot(test_predict, label='Предсказанные значения')

    # y = [i ** 2 for i in range(101)]
    # plot1.plot(y)

    plot1.set_xlabel('День')
    plot1.set_ylabel('Цена закрытия')
    plot1.set_title(
        'Прогнозирование цены закрытия криптовалюты\nMSE на тестовой выборке с использование модели: {:.4f}'.format(
            test_loss))
    plot1.legend()

    # Create a Tkinter canvas containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    # Place the canvas on the Tkinter frame
    canvas.get_tk_widget().pack()
    # Create a Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas, frame)
    toolbar.update()
    # Place the toolbar on the Tkinter frame
    canvas.get_tk_widget().pack()

    out = get_result(test_y, test_predict)
    insert_result(results_box, out)

def plot_predict(frame, results_box, clicked):
    # Clear the frame
    for widget in frame.winfo_children():
        widget.destroy()

    # get code crypto
    code = clicked.get()
    print(code)

    # Create a figure
    fig = Figure(figsize=(6, 4), dpi=100)

    # Add a subplot
    all_prices, data_visual, result_predict = before_predict(code)
    plot1 = fig.add_subplot(111)
    # Plot data

    plot1.plot(all_prices, label='Предсказанные значения', color='orange')
    plot1.plot(data_visual, label='Истинные значения')


    plot1.set_xlabel('День')
    plot1.set_ylabel('Цена закрытия')
    plot1.set_title(
        'Прогнозирование цены закрытия криптовалюты на следующие 30 дней')
    plot1.legend()

    # Create a Tkinter canvas containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    # Place the canvas on the Tkinter frame
    canvas.get_tk_widget().pack()
    # Create a Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas, frame)
    toolbar.update()
    # Place the toolbar on the Tkinter frame
    canvas.get_tk_widget().pack()

    out = get_predict(data_visual[-1][0], result_predict)
    insert_result(results_box, out)

def app_interfaces():
    # Create the main window
    root = tk.Tk()
    root.title("Справочное приложение для криптовалютных инвестиций")
    root.geometry('800x730')
    root.resizable(False, False)
    root.config(bg="skyblue")  # specify background color

    # Create a frame for the load button and file names box
    left_frame_1 = Frame(root, width=200, height=200, bg='grey')
    left_frame_1.grid(row=0, column=0, padx=10, pady=5)

    # Create a frame for the load button and file names box
    left_frame_2 = Frame(root, width=200, height=200, bg='grey')
    left_frame_2.grid(row=1, column=0, padx=10, pady=10)

    # Create a frame for the load button and file names box
    left_frame_3 = Frame(root, width=200, height=400, bg='grey')
    left_frame_3.grid(row=2, column=0, padx=10, pady=10)

    right_frame_1 = Frame(root, width=600, height=200, bg='grey')
    right_frame_1.grid(row=0, column=1, padx=10, pady=5)

    right_frame_2 = Frame(root, width=600, height=450, bg='grey')
    right_frame_2.grid(row=1, column=1, padx=10, pady=5)

    right_frame_3 = Frame(root, width=600, height=400, bg='grey')
    right_frame_3.grid(row=2, column=1, padx=10, pady=5)

    # Select box to select Crypto code
    options = ['BTC', 'LTC', 'ETH']
    clicked = StringVar()
    clicked.set('BTC')
    dropbox = OptionMenu(right_frame_1, clicked, *options)
    dropbox.pack(side=tk.TOP, padx=5, pady=5)
    max_length = len(max(options, key=len))
    dropbox.config(width=max_length)

    label = Label(left_frame_1, text="Select Crypto Code", bg='skyblue')
    label.pack(side=tk.TOP, fill="x")

    load_docx_button = tk.Button(left_frame_2, text="Train", command=lambda: training(right_frame_2, analysis_results_box, clicked))
    load_docx_button.pack(side=tk.TOP, padx=5, pady=5)

    # Create a box for displaying analysis results
    analysis_results_box = Text(right_frame_3, height=11, width=74)
    analysis_results_box.config(state=tk.DISABLED)
    analysis_results_box.pack(side=tk.TOP, padx=5, pady=5)

    # Create a button for performing analysis
    analysis_button = tk.Button(left_frame_3, text="Analyze", command=lambda: predict(right_frame_2, analysis_results_box, clicked))
    analysis_button.pack(side=tk.TOP, padx=5, pady=5)

    root.mainloop()

if __name__ == '__main__':
    app_interfaces()
