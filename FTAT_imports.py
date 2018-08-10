import pandas as pd
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

# parameter map class
class ParameterMap:
    def __init__(self, sub_sampled_data, color, title):
        # parameter map and its tool tip
        self.sub_sampled_data = sub_sampled_data
        self.sc_x = DateScale()
        self.sc_y = LinearScale()

        self.ax_x = Axis(scale=self.sc_x, grid_lines='dashed', label='Data')
        self.ax_y = Axis(scale=self.sc_y, orientation='vertical', grid_lines='dashed')

        self.line = Lines(x= self.sub_sampled_data.index.values, y=[], scales={'x': self.sc_x, 
                                                                                    'y': self.sc_y}, stroke_width=1)
        self.fig_tooltip = Figure(marks=[self.line], axes=[self.ax_x, self.ax_y], 
                                  layout=Layout(max_height='90%', max_width='90%'),
                                 stroke_width = 1)

        self.map_names = list(self.sub_sampled_data.columns)
        self.map_codes = [i for i in range(len(self.map_names))]
        self.map = MarketMap(names=self.map_names,      
                               layout=Layout(min_width='800px', min_height='100px'),
                                 enable_hover=False,
                                 tooltip_widget=self.fig_tooltip)

        self.map.colors = [color]
        self.map.font_style = {'font-size': '09px', 'fill':'white'}
        self.map.title = title
        self.map.title_style = {'fill': 'Red'}

        self.hovered_symbol = ''
        
    def hover_handler(self, _, content):
        self.symbol = content.get('data', '')

        if(self.symbol != self.hovered_symbol):
            self.hovered_symbol = self.symbol
            #if(sub_sampled_data.get(hovered_symbol) is not None):
            if(self.sub_sampled_data[self.hovered_symbol] is not None):
                self.line.y = self.sub_sampled_data[self.hovered_symbol].values
                #test_fig_tooltip.title = content.get('ref_data', {}).get('Name', '')

########
class LinePlot:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        # scales
        self.xs = DateScale()
        self.ys = LinearScale()

        # mark or figure type
        self.line = Lines(x=self.x_data, y=self.y_data, scales={'x': self.xs, 'y': self.ys}, stroke_width=1)

        # axis
        self.xax = Axis(scale=self.xs, label='Time', grids='on', tick_format='%H:%M:%S.%L')
        self.yax = Axis(scale=self.ys, orientation='vertical', grids='on', grid_lines='dashed')

        # canvas or actual figure                                                                     ##################
        self.fig = Figure(marks=[self.line], axes=[self.xax, self.yax], layout=Layout(width = '80%'))   
        ##################
        

########

#THIS DOES NOT WORK
#updating scales and axis does not update graph
#class CrossPlot(LinePlot):
#    def __init__(self, x_data, y_data):
#        super().__init__(x_data, y_data)
#        self.xs = LinearScale()
#        self.xax = Axis(scale=self.xs, orientation= 'horizontal', side='bottom',
#                        grids='on', grid_lines='dashed')
#        self.fig = Figure(marks=[self.line], axes=[self.xax, self.yax], layout=Layout(width = '80%')) ##need to redefine the fig to activate new xax

########
#had to redefine plot from scratch
#because redefining scale and axis did not work
class CrossPlot:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        # scales
        self.xs = LinearScale()
        self.ys = LinearScale()

        # mark or figure type
        self.line = Lines(x=self.x_data, y=self.y_data, scales={'x': self.xs, 'y': self.ys}, stroke_width=1)

        # axis
        self.xax = Axis(scale=self.xs, orientation='horizontal', grids='on', grid_lines='dashed')
        self.yax = Axis(scale=self.ys, orientation='vertical', grids='on', grid_lines='dashed')

        # canvas or actual figure
        self.fig = Figure(marks=[self.line], axes=[self.xax, self.yax], layout=Layout(width = '80%'))
#######


class ZoomPlot(LinePlot):
    def __init__(self, x_data, y_data):
        super().__init__(x_data, y_data)

        # objects that define the PanZoom for each case
        self.pz = PanZoom(scales={'y': [self.ys], 'x': [self.xs] })
        self.pzx = PanZoom(scales={ 'x': [self.xs] })
        self.pzy = PanZoom(scales={'y': [self.ys] })

        self.buttonWidth = '60px'
        self.zoom_interacts = widgets.ToggleButtons(options=OrderedDict([
                                                        (' ', None),
                                                        ('xy ', self.pz), 
                                                        ('x ', self.pzx), 
                                                        ('y ', self.pzy)]),
                                                        icons = ["stop", "arrows", "arrows-h", "arrows-v"]
            )
        self.zoom_interacts.style.button_width = self.buttonWidth   # ipywidgets 7.0 and above

        self.reset_zoom_button = widgets.Button(
            description='',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Reset zoom',
            icon='arrows-alt',
            layout = Layout(width = self.buttonWidth)
        )
        self.reset_zoom_button.on_click(self.resetZoom)

        # linking both
        self.zoom_link = link((self.zoom_interacts, 'value'), (self.fig, 'interaction'))

    def resetZoom(self, new):
        self.ys.min=None
        self.ys.max=None

        # Fix the x axes.  
        self.xs.min=min([mark.x.min() for mark in self.fig.marks])
        self.xs.max=max([mark.x.max() for mark in self.fig.marks])

class AnalysisPlot(ZoomPlot):
    def __init__(self, x_data, y_data, time_slices_db):
        super().__init__(x_data, y_data)
        self.x_fitted_data = x_data
        self.y_fitted_data = y_data
        self.x_data_slice_min = x_data.min()
        self.x_data_slice_max = x_data.max()
        #self.last_saved_slice = next(iter(time_slices_db.values()))
        self.fitted_line = Lines(x=self.x_fitted_data, y=self.y_fitted_data, scales={'x': self.xs, 'y': self.ys},
                                 stroke_width=1, colors=['red'], line_style='dashed')
        self.fig = Figure(marks=[self.line, self.fitted_line], axes=[self.xax, self.yax], 
                          title_style={'font-size': '14px','fill': 'DarkOrange'}, 
                          title='No Parameter Selected',
                          layout=Layout(width = '80%'))
        self.xs.min=min([mark.x.min() for mark in self.fig.marks])
        self.xs.max=max([mark.x.max() for mark in self.fig.marks])
        self.fit_statistics = widgets.HTML(
                                        value="Empty <b>Empty</b>",
                                        placeholder='Poly Coefs',
                                        description='Poly Coefs',
                                    )
        self.fit_statistics.value = 'Empty'
        self.plot_toolbar = Toolbar(figure=self.fig)
        
        
    def resetZoom(self, new):                       
        self.xs.min=min([mark.x.min() for mark in self.fig.marks])
        self.xs.max=max([mark.x.max() for mark in self.fig.marks])
        self.ys.min=min([mark.y.min() for mark in self.fig.marks])
        self.ys.max=max([mark.y.max() for mark in self.fig.marks])
        
        
    def update_plot(self, current_plot_data, parameter_list, slice_start, slice_end, poly_degree):
        if parameter_list: #this means the list is not empty
            self.xs.min = slice_start #analysis_TP_dd.value
            self.xs.max = slice_end
            slice_start_index = current_plot_data.index.searchsorted(self.xs.min)
            initial_value = current_plot_data.index[slice_start_index].timestamp()
            xdata = current_plot_data.iloc[(current_plot_data.index >= self.xs.min) & 
                                           (current_plot_data.index <= self.xs.max)].index.astype(int).astype(float) \
                                           - (initial_value*1e9)
            xdata = xdata / 1e9  ##
            # ydata is going to be the first selected parameter, for now, thus the "0" below.
            ydata = current_plot_data.iloc[(current_plot_data.index >= self.xs.min) & 
                                           (current_plot_data.index <= self.xs.max), 0].values
            linefit = np.polyfit(xdata, ydata, poly_degree)
            fittedx = np.linspace(xdata[0], xdata[-1], len(xdata))

            fittedy = np.polyval(linefit, fittedx)
            #self.fitted_line.x = pd.to_datetime(fittedx+(initial_value*1e9))
            self.fitted_line.x = pd.to_datetime((fittedx*1e9)+(initial_value*1e9))  ##
            self.fitted_line.y = fittedy
            self.ys.min=ydata.min(axis=0)
            self.ys.max=ydata.max(axis=0)
            stats_string = 'Highest to lowest expoents </p>'
            #stats_string = str(linefit)
            for x in linefit:
                #stats_string = stats_string + '<p>{:.6f}'.format(x) + '</p>'
                stats_string = stats_string + str(x) + '</p>'
            stats_string = stats_string + 'Min X= {:.2f}'.format(xdata[0]) + '</p>'
            stats_string = stats_string + 'Max X= {:.2f}'.format(xdata[-1]) + '</p>'
            self.fit_statistics.value = stats_string

        
        

class sliceSelectDialog():
    def __init__(self, current_plot_data):
        
        # time slice selection buttons and logic
        self.startTS_box_title = Label()
        self.startTS_box_title.value = "Start of TS:"
        self.start_hour_box = widgets.BoundedIntText(
            value=current_plot_data.index[0].hour,
            min=current_plot_data.index[0].hour,
            max=current_plot_data.index[-1].hour,
            step=1,
            description='Hour:',
            disabled=False,
            layout=Layout(width='140px')
        )

        self.start_minute_box = widgets.BoundedIntText(
            value=0,
            min=0,
            max=59,
            step=1,
            description='Minute:',
            disabled=False,
            layout=Layout(width='140px')
        )

        self.start_second_box = widgets.BoundedIntText(
            value=0,
            min=0,
            max=59,
            step=1,
            description='Second:',
            disabled=False,
            layout=Layout(width='140px')
        )

        self.endTS_box_title = Label()
        self.endTS_box_title.value = "_____End of TS:"
        self.end_hour_box = widgets.BoundedIntText(
            value=current_plot_data.index[0].hour,
            min=current_plot_data.index[0].hour,
            max=current_plot_data.index[-1].hour,
            step=1,
            description='Hour:',
            disabled=False,
            layout=Layout(width='140px')
        )

        self.end_minute_box = widgets.BoundedIntText(
            value=0,
            min=0,
            max=59,
            step=1,
            description='Minute:',
            disabled=False,
            layout=Layout(width='140px')
        )

        self.end_second_box = widgets.BoundedIntText(
            value=0,
            min=0,
            max=59,
            step=1,
            description='Second:',
            disabled=False,
            layout=Layout(width='140px')
        )
        
        self.boxes_item_layout = Layout(height='', min_width='40px')
        self.slice_box_items = [self.startTS_box_title, self.start_hour_box, self.start_minute_box, self.start_second_box,
                                self.endTS_box_title, self.end_hour_box, self.end_minute_box, self.end_second_box]
        self.boxes_layout = Layout(overflow_x='scroll',
                    border='3px solid black',
                    width='1000px',
                    height='',
                    flex_direction='row',
                    display='flex')

class DataSliceSelect():
    def __init__(self, time_slices_db=None):
        self.analysisTPdd = widgets.Dropdown(
        options=time_slices_db,
        description='TP#:',
        value=None,
        disabled=False
        )

        self.analysisTPDescBox = widgets.Label(
            value=''
        )

        self.analysisPolyOrderdd = widgets.Dropdown(
            options=[1,2,3,4],
            description='Poly Deg',
            value=1,
            disabled=True
        )
        
    def update_TPdd(self, time_slices_db):
        options_copy = time_slices_db.copy()
        self.analysisTPdd.options = options_copy

class SimpleZoomSlider():
    def __init__(self, plot):
        #self.minval = int(time.mktime(datetime.utcfromtimestamp(np.datetime64(plot.x_data.min()).astype('O')/1e9).timetuple()))
        #self.maxval = int(time.mktime(datetime.utcfromtimestamp(np.datetime64(plot.x_data.max()).astype('O')/1e9).timetuple()))
        self.minval = int(time.mktime(plot.xs.min.timetuple()))
        self.maxval = int(time.mktime(plot.xs.max.timetuple()))
        self.delta_time_int = self.maxval-self.minval
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
        #self.minval = int(time.mktime(datetime.utcfromtimestamp(np.datetime64(plot.fitted_line.x.min()).astype('O')/1e9).timetuple()))
        #self.maxval = int(time.mktime(datetime.utcfromtimestamp(np.datetime64(plot.fitted_line.x.max()).astype('O')/1e9).timetuple()))
        
        #self.minval = int(time.mktime(plot.xs.min.timetuple()))
        #self.maxval = int(time.mktime(plot.xs.max.timetuple()))
        
        self.minval =  int(time.mktime(datetime.utcfromtimestamp(np.datetime64(plot.x_data_slice_min).astype('O')/1e9).timetuple()))
        self.maxval =  int(time.mktime(datetime.utcfromtimestamp(np.datetime64(plot.x_data_slice_max).astype('O')/1e9).timetuple()))
        
        self.delta_time_int = self.maxval-self.minval
        plot.xs.min = datetime.fromtimestamp(self.minval + self.zoom_slider.value[0]/100*self.delta_time_int)
        plot.xs.max = datetime.fromtimestamp(self.minval + self.zoom_slider.value[1]/100*self.delta_time_int)
        #  TODO - when slider is changed, recalculate the fitted line
        #  maybe change the plot method so that we do not need to pass the current_plot_data anymore
        


class StripChart(object):
    def __init__(self, dataframe, selection_map, ts_source_plot, ts_title):
        #we pass the time slice source plot instead of the time slice in the db dictionary
        #because if the user changes the start/end with the slider, we want the strip charts to be in sync
        self.dataframe = dataframe
        self.selection_map = selection_map
        self.ts_source_plot = ts_source_plot
        self.ts_title = ts_title
        #self.path = os.getcwd()
        self._update_charts()
        self.msg = ''
        self.matplotlib_button_clicked = False
        
    def _update_charts(self):
        
        if self.ts_source_plot.xs.min:
            self.slice_start = np.datetime64(self.ts_source_plot.xs.min)
            self.slice_end = np.datetime64(self.ts_source_plot.xs.max)
            self.msg = ''
        else:
            self.slice_start = self.ts_source_plot.x_data_slice_min
            self.slice_end = self.ts_source_plot.x_data_slice_max
            self.msg = 'Select Time Slice Above!!!'
        
        self.slice_start_index = self.dataframe.index.searchsorted(self.slice_start)
        initial_value = self.dataframe.index[self.slice_start_index].timestamp()
        self.strip_chart_x_data = self.dataframe.iloc[(self.dataframe.index >= self.slice_start) & 
                                       (self.dataframe.index <= self.slice_end)].index
            
       
    def widget(self):
        box = widgets.VBox()
        self._update(box)
        return box
    
    def _update(self, box):
        
        def on_click_update_SC(b):
            self.matplotlib_button_clicked = False
            self._update_charts()
            self._update(box)
        
        def on_click_final_plot(b):
            self.matplotlib_button_clicked = True
            self._update_charts()
            self._update(box)
        
        plot_button = widgets.Button(description='Draft Plot', background_color='#d0d0ff', layout=Layout(width='50%', height='35px'))
        plot_button.on_click(on_click_update_SC)
        matplotlib_plot_button = widgets.Button(description='Final Plot', background_color='#1f9b1d', layout=Layout(width='50%', height='35px'))
        matplotlib_plot_button.on_click(on_click_final_plot)
        plotlist = []
        if self.matplotlib_button_clicked:
            
            counter = 1
            #clear_output()
            outbox = Output()
            myfig = plt.figure(figsize = (16,(len(self.selection_map.map.selected)*4)))
            plt.subplots_adjust(hspace = 0.0)
            plt.title(self.ts_title)
            with outbox:
                for selection in self.selection_map.map.selected:
                    strip_chart_y_data = self.dataframe.iloc[(self.dataframe.index >= self.slice_start) & 
                                               (self.dataframe.index <= self.slice_end)][selection].values
                    ax = myfig.add_subplot(len(self.selection_map.map.selected), 1, counter)

                    ax.plot(self.strip_chart_x_data, strip_chart_y_data)
                    plt.ylabel(selection)
                    if counter < len(self.selection_map.map.selected):
                        ax.axes.get_xaxis().set_ticks([])
                    counter += 1
            ab = myfig.get_axes()
            ac = ab[0]
            ac.tick_params(bottom=False, top = False, left = False, right = False)
            ac.tick_params(labelbottom=False, labeltop = False, labelleft = False, labelright = False)
                

            # Time Slices Analysis Section
            strip_chart_items = [plot_button, matplotlib_plot_button]
            strip_chart_items.append(outbox)
            strip_chart_items.insert(1, widgets.Label(self.msg))
            box.children = strip_chart_items
            
        else:
            
            counter = 0
            for selection in self.selection_map.map.selected:
                strip_chart_y_data = self.dataframe.iloc[(self.dataframe.index >= self.slice_start) & 
                                           (self.dataframe.index <= self.slice_end)][selection].values
                plotlist.append(LinePlot(self.strip_chart_x_data, strip_chart_y_data))
                plotlist[counter].fig.title = selection
                counter += 1

            # Time Slices Analysis Section
            strip_chart_items = [plot_button, matplotlib_plot_button]
            for plots in plotlist:
                plots.xs.min = None
                plots.xs.max = None
                plots.ys.min = None
                plots.ys.max = None
                plots.xax.tick_format = '%M:%S.%L'
                strip_chart_items.append(plots.fig)
            strip_chart_items.insert(1, widgets.Label(self.msg))
            box.children = strip_chart_items