import datetime
import PySimpleGUI as sg
from PytorchForecasting import TFT
import matplotlib.pyplot as plt
import io
import base64

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

sg.theme("LightBlue")

layout = [
    [sg.Text("Stock Prediction Input")],
    [sg.Text("Enter Stock Ticker:"), sg.InputText(key='-TICKER-', default_text='AAPL')],
    [sg.Text("Select Start Date for Reference Data (YYYY-MM-DD):"), sg.Input(key='-START-', default_text='2015-01-02')],
    [sg.Text("Select End Date for Reference Data (YYYY-MM-DD):"), sg.Input(key='-END-', default_text='2019-12-31')],
    [sg.Text("Select Start Date for Prediction (YYYY-MM-DD):"), sg.Input(key='-PSTART-', default_text='2020-01-02')],
    [sg.Text("Select End Date for Prediction (YYYY-MM-DD):"), sg.Input(key='-PEND-', default_text='2020-12-31')],
    [sg.Button("Run Prediction")],
    [sg.Text("Prediction Results:")],
    [sg.Canvas(key='-CANVAS-')],
    [sg.Button("Exit")]
]

window = sg.Window("TFT Stock Prediction", layout, finalize=True)

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    if event == "Run Prediction":
        try:
            ticker = values['-TICKER-']
            start_date = datetime.datetime.strptime(values['-START-'], "%Y-%m-%d").date()
            end_date = datetime.datetime.strptime(values['-END-'], "%Y-%m-%d").date()
            pstart = datetime.datetime.strptime(values['-PSTART-'], "%Y-%m-%d").date()
            pend = datetime.datetime.strptime(values['-PEND-'], "%Y-%m-%d").date()

            tft_inst = TFT()
            tft_inst.load_data_yfinance(ticker, start_date, end_date, pstart, pend)
            tft_inst.create_tft_model()
            tft_inst.train()
            fig = tft_inst.evaluate_and_plot()

            if fig is None:
                sg.popup("No figure was returned. Please check console logs and date ranges.")
            else:
                fig_canvas_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)

        except Exception as e:
            sg.popup("Error", str(e))

window.close()
