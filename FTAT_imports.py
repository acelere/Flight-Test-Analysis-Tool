import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import numpy as np
import sys
from datetime import datetime
import time

from collections import OrderedDict

from bqplot import *
from bqplot.interacts import (
    FastIntervalSelector, IndexSelector, BrushIntervalSelector,
    BrushSelector, MultiSelector, LassoSelector, PanZoom, HandDraw
)
from bqplot.market_map import MarketMap
from traitlets import link

from ipywidgets import ToggleButtons, VBox, HBox, HTML, Box, Output
from ipywidgets import widgets
from ipywidgets import Label, Layout, Style
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import display, HTML

import matplotlib
import matplotlib.pyplot as plt

import os


#functions
def data_slicer(data_stream, slice_start_time, slice_end_time):
    #helper function to return the sliced data
    #if start/stop outside range, return full data
    #data_stream is a pandas df
    # start/stop times are datetime64 values
    #the reason to have the selected_slice is growth capability
    #in the future, I want to maybe use these times to slice a movie
    #therefore I cannot store/lookup the index of the data. Need times.
    
    if slice_start_time < data_stream.index.values[0]:
        slice_start_time = data_stream.index.values[0]
    if (slice_end_time > data_stream.index.values[-1]) or (slice_end_time <= slice_start_time) :
        slice_end_time = data_stream.index.values[-1]
    slice_start_index = data_stream.index.searchsorted(slice_start_time)
    slice_end_index = data_stream.index.searchsorted(slice_end_time)
    sliced_data = data_stream.iloc[slice_start_index:slice_end_index]
    selected_slice = [sliced_data.index.values[0], sliced_data.index.values[-1]]
    return sliced_data, selected_slice

def update_analysis_plot(current_plot_data, slicemap_map_selected, plot_object, poly_order, tz_slider, zoom_slider):
    
    delta_time = np.timedelta64(np.datetime64(plot_object.x_data_slice_max, 'us') - np.datetime64(plot_object.x_data_slice_min, 'us'))
    new_xs_min = (np.datetime64(plot_object.x_data_slice_min, 'us') + zoom_slider.zoom_slider.value[0]/100*delta_time).astype(datetime)
    new_xs_max = (np.datetime64(plot_object.x_data_slice_min, 'us') + zoom_slider.zoom_slider.value[1]/100*delta_time).astype(datetime)
    
    plot_object.update_plot(current_plot_data, slicemap_map_selected, 
                              np.datetime64(new_xs_min, 'us'),
                              np.datetime64(new_xs_max, 'us'),
                              poly_order, 
                              tz_slider)

# parameter map class
class ParameterMap:
    def __init__(self, sub_sampled_data, color, title):
        # parameter map and its tool tip
        self.sub_sampled_data = sub_sampled_data
        self.map_names = list(self.sub_sampled_data.columns)
        self.map_codes = [i for i in range(len(self.map_names))]
        
        self.map = MarketMap(names=self.map_names,      
                               layout=Layout(min_width='50px', min_height='70px'),
                                 enable_hover=False, cols=3,
                            map_margin={'top':50, 'bottom':0, 'left':0, 'right':25})

        self.map.colors = [color]
        self.map.font_style = {'font-size': '10px', 'fill':'white'}
        self.map.title = title
        self.map.title_style = {'fill': 'Red'}

class sliceSelectDialog():
    def __init__(self, current_plot_data):
        
        # time slice selection buttons and logic
        self.startTS_box_title = Label()
        self.startTS_box_title.value = "Start"
        self.start_hour_box = widgets.BoundedIntText(
            value=current_plot_data.index[0].hour,
            min=current_plot_data.index[0].hour,
            max=current_plot_data.index[-1].hour,
            step=1,
            description='H',
            disabled=False,
            layout=Layout(width='140px')
        )

        self.start_minute_box = widgets.BoundedIntText(
            value=0,
            min=0,
            max=59,
            step=1,
            description='M',
            disabled=False,
            layout=Layout(width='140px')
        )

        self.start_second_box = widgets.BoundedIntText(
            value=0,
            min=0,
            max=59,
            step=1,
            description='S',
            disabled=False,
            layout=Layout(width='140px')
        )

        self.endTS_box_title = Label()
        self.endTS_box_title.value = "_____End - H:M:S"
        self.end_hour_box = widgets.BoundedIntText(
            value=current_plot_data.index[0].hour,
            min=current_plot_data.index[0].hour,
            max=current_plot_data.index[-1].hour,
            step=1,
            description='H',
            disabled=False,
            layout=Layout(width='140px')
        )

        self.end_minute_box = widgets.BoundedIntText(
            value=0,
            min=0,
            max=59,
            step=1,
            description='M',
            disabled=False,
            layout=Layout(width='140px')
        )

        self.end_second_box = widgets.BoundedIntText(
            value=0,
            min=0,
            max=59,
            step=1,
            description='S',
            disabled=False,
            layout=Layout(width='140px')
        )
        
        self.boxes_item_layout = Layout(height='', min_width='40px')
        self.slice_box_items = [self.start_hour_box, self.start_minute_box, self.start_second_box,
                                self.end_hour_box, self.end_minute_box, self.end_second_box]
        self.boxes_layout = Layout(overflow_x='scroll',
                    border='1px solid black',
                    height='',
                    flex_direction='row',
                    display='flex')
        self.slice_box = HBox(self.slice_box_items)
        
    def update_values(self, start, end):
        #start and stop are numpy datetime64 objects
        self.start_hour_box.value = start.astype(object).hour
        self.start_minute_box.value = start.astype(object).minute
        self.start_second_box.value = start.astype(object).second
        
        self.end_hour_box.value = end.astype(object).hour
        self.end_minute_box.value = end.astype(object).minute
        self.end_second_box.value = end.astype(object).second
######

class DataSliceSelect():
    def __init__(self, time_slices_db=None):

        self.analysisTPDescBox = widgets.Label(
            value=''
        )

        self.analysisPolyOrderdd = widgets.Dropdown(
            options=[1,2,3,4],
            description='Poly Deg',
            value=1,
            disabled=True
        )

class SimpleZoomSlider():
    def __init__(self, plot):

        self.minval = np.datetime64(plot.x_data_slice_min, 'us')
        self.maxval = np.datetime64(plot.x_data_slice_max, 'us')

        self.delta_time_int = np.timedelta64(self.maxval-self.minval)
        self.my_slider_layout = Layout(max_width='100%', width='80%', height='75px')
        self.zoom_slider= widgets.IntRangeSlider(
            value=[0, 100],
            step=1,
            description='Zoom:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            layout=self.my_slider_layout
        )
     
    
    def updateScale(self, plot):
        self.minval = np.datetime64(plot.xs.min, 'us') #type datetime.datetime - from bqplot
        self.maxval = np.datetime64(plot.xs.max, 'us')
        plot.x_data_slice_min = self.minval
        plot.x_data_slice_max = self.maxval

        self.delta_time_int = np.timedelta64(self.maxval-self.minval)


########
class LinePlot:
    def __init__(self, x_data, y_data):
        self.x_data = x_data #this is comming in as np.datetime64
        self.y_data = y_data
        # scales
        self.xs = DateScale()
        self.ys = LinearScale()

        # mark or figure type
        self.line = Lines(x=self.x_data, y=self.y_data, scales={'x': self.xs, 'y': self.ys},
                              selected_style={'opacity':'1'}, unselected_style={'opacity':'0.2'},
                              stroke_width=1)
        
        
        # axis
        # for axis format, see:
        #    https://bqplot.readthedocs.io/en/latest/_generate/bqplot.axes.Axis.html#bqplot.axes.Axis.tick_format
        #    https://github.com/d3/d3-3.x-api-reference/blob/master/Time-Formatting.md
        self.xax = Axis(scale=self.xs, label='Time', grids='on', tick_format='%H:%M:%S', tick_rotate=30)
        self.yax = Axis(scale=self.ys, orientation='vertical', grids='on', grid_lines='dashed')

        self.fig = Figure(marks=[self.line], axes=[self.xax, self.yax], layout=Layout(width = '80%'))
########


########
class LinePlotBrush(LinePlot):
    def __init__(self, x_data, y_data):
        super().__init__(x_data, y_data)
    
        self.brushintsel = BrushIntervalSelector(scale=self.xs)

        self.fig = Figure(marks=[self.line], axes=[self.xax, self.yax], layout=Layout(width = '80%'), interaction=self.brushintsel)
########

########
class AnalysisPlot(LinePlotBrush):
    def __init__(self, x_data, y_data, tz_slider):
        super().__init__(x_data, y_data)
        self.x_fitted_data = x_data
        self.y_fitted_data = y_data
        self.x_data_slice_min = x_data.min()
        self.x_data_slice_max = x_data.max()
        self.fitted_line = Lines(x=self.x_fitted_data, y=self.y_fitted_data, scales={'x': self.xs, 'y': self.ys},
                                 stroke_width=1, colors=['red'], line_style='dashed')
        self.fig = Figure(marks=[self.line, self.fitted_line], axes=[self.xax, self.yax], 
                          title_style={'font-size': '14px','fill': 'DarkOrange'}, 
                          title='No Parameter Selected',
                          layout=Layout(width = '80%'), 
                          interaction=self.brushintsel)
        self.xs.min=min([mark.x.min() for mark in self.fig.marks]) + np.timedelta64(tz_slider, 'h') #this is datetime.datetime internally to bqplot
        self.xs.max=max([mark.x.max() for mark in self.fig.marks]) + np.timedelta64(tz_slider, 'h')
        self.fit_statistics = widgets.HTML(
                                        value="Empty <b>Empty</b>",
                                        placeholder='Poly Coefs',
                                        description='Poly Coefs',
                                    )
        self.fit_statistics.value = 'Empty'
        
        
    def update_plot(self, current_plot_data, parameter_list, slice_start, slice_end, poly_degree, tz_slider):
        '''
        current_plot_data
        parameter_list
        slice_start: numpy.datetime64, 'us' - from time_slices_db
        slice_end: numpy.datetime64, 'us'
        poly_degrees: int
        '''
        if parameter_list: #this means the list is not empty
            self.xs.min = slice_start + np.timedelta64(tz_slider, 'h') #I can pass a np.datetime64, bqplot converts internally to datetime.datetime
            self.xs.max = slice_end + np.timedelta64(tz_slider, 'h')
            slice_start_index = current_plot_data.index.searchsorted(slice_start)
            initial_value = current_plot_data.index[slice_start_index].timestamp()
            xdata = current_plot_data.iloc[(current_plot_data.index >= slice_start) & 
                                           (current_plot_data.index <= slice_end)].index.astype(int).astype(float) \
                                           - (initial_value*1e9)
            xdata = xdata / 1e9  ##
            # ydata is going to be the first selected parameter, for now, thus the "0" below.
            ydata = current_plot_data.iloc[(current_plot_data.index >= slice_start) & 
                                           (current_plot_data.index <= slice_end), 0].values
            linefit = np.polyfit(xdata, ydata, poly_degree)
            fittedx = np.linspace(xdata[0], xdata[-1], len(xdata))

            fittedy = np.polyval(linefit, fittedx)

            self.fitted_line.x = pd.to_datetime((fittedx*1e9)+(initial_value*1e9))  ##
            self.fitted_line.y = fittedy
            self.ys.min=ydata.min(axis=0)
            self.ys.max=ydata.max(axis=0)
            stats_string = 'Highest to lowest expoents </p>'

            for x in linefit:
                stats_string = stats_string + '<p>{:.6f}'.format(x) + '</p>'
                #stats_string = stats_string + str(x) + '</p>'
            stats_string = stats_string + 'Min X= {:.2f}'.format(xdata[0]) + '</p>'
            stats_string = stats_string + 'Max X= {:.2f}'.format(xdata[-1]) + '</p>'
            stats_string = stats_string + 'Avg Y= {:.2f}'.format(ydata.mean()) + '</p>'
            stats_string = stats_string + 'Std Y= {:.2f}'.format(ydata.std()) + '</p>'
            self.fit_statistics.value = stats_string
##########




class StripChart(object):
    def __init__(self, dataframe, selection_map, ts_db, ts_db_radio):
        self.dataframe = dataframe
        self.selection_map = selection_map
        self.ts_db = ts_db
        self.ts_db_radio = ts_db_radio
        if ts_db_radio.value:
            self.msg = ts_db.get(ts_db_radio.value)[0]
            self.slice_start = ts_db.get(ts_db_radio.value)[1]
            self.slice_end = ts_db.get(ts_db_radio.value)[2]
        else:
            self.msge = 'Test Point not set yet.'
            self.slice_start = dataframe.index.min()
            self.slice_end = dataframe.index.max()
        #self.path = os.getcwd()
        self._update_charts()
        self.matplotlib_button_clicked = False
        
    def _update_charts(self):
               
        if self.ts_db_radio.value:
            self.msg = self.ts_db.get(self.ts_db_radio.value)[0]
            self.slice_start = self.ts_db.get(self.ts_db_radio.value)[1]
            self.slice_end = self.ts_db.get(self.ts_db_radio.value)[2]
        else:
            self.msg = 'Test Point not set yet.'
            self.slice_start = self.dataframe.index.min()
            self.slice_end = self.dataframe.index.max()
        self.slice_start_index = self.dataframe.index.searchsorted(self.slice_start)
        initial_value = self.dataframe.index[self.slice_start_index].timestamp()
        self.strip_chart_x_data = self.dataframe.iloc[(self.dataframe.index >= self.slice_start) & 
                                       (self.dataframe.index <= self.slice_end)].index
            
       
    def widget(self):
        box = widgets.VBox()
        self._update(box)
        return box
    
    def _update(self, box):
        
        def on_click_final_plot(b):
            self.matplotlib_button_clicked = True
            self._update_charts()
            self._update(box)
        
        matplotlib_plot_button = widgets.Button(description='Final Plot', background_color='#1f9b1d', layout=Layout(width='50%', height='35px'))
        matplotlib_plot_button.on_click(on_click_final_plot)
        plotlist = []
        if self.matplotlib_button_clicked:
            plt.ioff()
            plt.clf()
            outbox = Output()
            counter = 1
            myfig = plt.figure(figsize = (16,(len(self.selection_map.map.selected)*4)))
            myfig.patch.set_edgecolor('w')
            plt.subplots_adjust(hspace = 0.0)
            with outbox:
                for selection in self.selection_map.map.selected:
                    strip_chart_y_data = self.dataframe.iloc[(self.dataframe.index >= self.slice_start) & 
                                               (self.dataframe.index <= self.slice_end)][selection].values
                    ax = myfig.add_subplot(len(self.selection_map.map.selected), 1, counter)
                    ax.plot(self.strip_chart_x_data, strip_chart_y_data)
                    plt.ylabel(selection)
                    plt.grid(True)
                    counter += 1
                print('To save figure, hold SHIFT and Right Click...Save image as...')
                plt.show();

            # Time Slices Analysis Section
            strip_chart_items = [matplotlib_plot_button]
            strip_chart_items.insert(1, widgets.Label(self.msg))
            strip_chart_items.append(outbox)
            box.children = strip_chart_items
        else:
            strip_chart_items = [matplotlib_plot_button]
            box.children = strip_chart_items

# ##########################
# File Browser Interface

# from:
# https://gist.github.com/DrDub/6efba6e522302e43d055



class FileBrowser(object):
    def __init__(self):
        self.path = os.getcwd()
        self._update_files()
        
    def _update_files(self):
        self.files = list()
        self.dirs = list()
        if(os.path.isdir(self.path)):
            for f in os.listdir(self.path):
                ff = self.path + "/" + f
                if os.path.isdir(ff):
                    self.dirs.append(f)
                    self.dirs.sort()
                else:
                    self.files.append(f)
                    self.files.sort()
        
    def widget(self):
        box = widgets.VBox()
        self._update(box)
        return box
    
    def _update(self, box):
        
        def on_click(b):
            if b.description == '..':
                self.path = os.path.split(self.path)[0]
            else:
                self.path = self.path + "/" + b.description
            self._update_files()
            self._update(box)
        
        buttons = []
        if self.files:
            button = widgets.Button(description='..', background_color='#d0d0ff', layout=Layout(width='70%', height='35px'))
            button.on_click(on_click)
            buttons.append(button)
        for f in self.dirs:
            button = widgets.Button(description=f, background_color='#d0d0ff', layout=Layout(width='70%', height='35px'))
            button.on_click(on_click)
            buttons.append(button)
        for f in self.files:
            button = widgets.Button(description=f, layout=Layout(width='70%', height='35px'))
            button.on_click(on_click)
            buttons.append(button)
        box.children = tuple([widgets.HTML("<h2>%s</h2>" % (self.path,))] + buttons)

