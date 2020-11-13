import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import numpy as np
import scipy.signal as signal
import sys
#from datetime import datetime
import datetime
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
import subprocess
from ipyfilechooser import FileChooser


#functions

def get_sub_sample_rate(raw_data, desired_sub_sample_rate):
    counter = 0
    start_sample = raw_data.index[0]
    next_sample = start_sample
    while (next_sample - start_sample <= np.timedelta64(1, 's')):
        counter = counter + 1
        next_sample = raw_data.index[counter]
    if desired_sub_sample_rate <= counter:
        sub_sample_rate = int(counter / desired_sub_sample_rate)
    else:
        sub_sample_rate = int(counter)
        desired_sub_sample_rate = counter
    return sub_sample_rate, counter


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
    if sliced_data.size > 0:
        selected_slice = [sliced_data.index.values[0], sliced_data.index.values[-1]]
    else:
        selected_slice = [slice_start_time, slice_end_time]
    return sliced_data, selected_slice



def lib_detail_plot_update_brush(*args, **kwargs):
    # get the selected TS from the brush
    # update the detail plot with the "zoomed-in" data
    
    current_plot_data = kwargs.get('cpt', None)
    selected_slice= kwargs.get('selsl', None)
    detail_plot = kwargs.get('detplt', None)
    slice_plot = kwargs.get('slcplt', None)
    slicebox = kwargs.get('slcbx', None)
    detail_plot_stats = kwargs.get('dtstats', None)
    
    if slice_plot.brushintsel.selected.size != 0:
            
        slice_start_time = slice_plot.brushintsel.selected[0]
        slice_end_time = slice_plot.brushintsel.selected[-1]

        sliced_data, selected_slice = data_slicer(current_plot_data, slice_start_time, slice_end_time)

        # check if the brush selector has at least 2 points...
        if sliced_data.shape[0] > 1:
            detail_plot.line.y = np.transpose(sliced_data).values
            detail_plot.line.x = sliced_data.index.values
            detail_plot_stats.value = sliced_data.describe().to_html()
            slicebox.update_values(slice_start_time, slice_end_time)


def lib_on_plot_button_clicked(b, **kwargs):
    
    current_plot_data = kwargs.get('cpt', None)
    detail_plot = kwargs.get('detplt', None)
    slicebox = kwargs.get('slcbx', None)
    detail_plot_stats = kwargs.get('dtstats', None)
    
    slice_start_index = 0

    current_start_time = current_plot_data.index[slice_start_index]
    
    new_start_time = current_start_time.replace(hour=slicebox.start_hour_box.value, 
                                                minute=slicebox.start_minute_box.value,
                                                second=slicebox.start_second_box.value)

    new_end_time = current_start_time.replace(hour=slicebox.end_hour_box.value, 
                                              minute=slicebox.end_minute_box.value,
                                              second=slicebox.end_second_box.value)
    
    slice_start_index = current_plot_data.index.searchsorted(new_start_time)
    slice_end_index = current_plot_data.index.searchsorted(new_end_time)
    
    sliced_data = current_plot_data.iloc[slice_start_index:slice_end_index]
    selected_slice = [sliced_data.index.values[0], sliced_data.index.values[-1]]
    
    # check if the brush selector has at least 2 points...
    if sliced_data.shape[0] > 1:
        detail_plot.line.y = np.transpose(sliced_data).values
        detail_plot.line.x = sliced_data.index.values
        #detail_plot.line.y = np.transpose(sliced_data).values
        detail_plot_stats.value = sliced_data.describe().to_html()
    
    


    
def lib_on_saveTS_button_clicked(b, **kwargs):
    
    current_plot_data = kwargs.get('cpt', None)
    detail_plot = kwargs.get('detplt', None)
    slice_plot = kwargs.get('slcplt', None)
    time_slices_db = kwargs.get('tsdb', None)
    TP_number_box = kwargs.get('tpnb', None)
    TS_message = kwargs.get('tsmsg', None)
    TP_desc_box = kwargs.get('tpdescb', None)
    TP_saved_dd = kwargs.get('tpsavdd', None)
    time_slices_db_radio = kwargs.get('tsdbrb', None)
    
    
    if str(TP_number_box.value) in time_slices_db:
        TS_message.value = 'TP# {} already in db, choose a different one'.format(TP_number_box.value)
    else:
        time_slices_db[str(TP_number_box.value)] = [TP_desc_box.value, detail_plot.line.x[0],\
                                                        detail_plot.line.x[-1]]

        TP_saved_dd.options = list(dict.keys(time_slices_db))
        time_slices_db_radio.options = list(dict.keys(time_slices_db))
        TS_message.value = 'added TP {}'.format(TP_number_box.value)
    
    
    
    
def lib_on_delTS_button_clicked(b, **kwargs):
    
    time_slices_db = kwargs.get('tsdb', None)
    TS_message = kwargs.get('tsmsg', None)
    TP_saved_dd = kwargs.get('tpsavdd', None)
    time_slices_db_radio = kwargs.get('tsdbrb', None)
    
    if TP_saved_dd.value in time_slices_db:
        TS_message.value = 'Deleted'
        del time_slices_db[TP_saved_dd.value]
        TP_saved_dd.options = list(dict.keys(time_slices_db))
        time_slices_db_radio.options = list(dict.keys(time_slices_db))
    else:
        TS_message.value = 'trying to delete nothing?'    
    


    
def lib_save_slices(**kwargs):
    
    tp_dict = kwargs.get('tsdb', None)
    save_all_parameters = kwargs.get('svall', None)
    raw_data = kwargs.get('rawdt', None)
    slicemap = kwargs.get('slcmap', None)
    save_feedback = kwargs.get('svfb', None)

    filepath = '/home/jovyan/work/'
    
    #creeate the TP data files
    filename_collector = []
    for key in tp_dict.keys():
        current_value = tp_dict[key]
        
        filename1 = filepath + 'TP_' +  str(key) + '_' + current_value[0] + '.csv'
        filename2 = filepath + 'TP_' +  str(key) + '_' + current_value[0] + '_' + 'stats' + '.csv'
        
        filename_collector.append(filename1)
        filename_collector.append(filename2)

        slice_start = current_value[1]
        slice_end = current_value[2]
        sliced_data = raw_data.iloc[(raw_data.index >= slice_start) & 
                                        (raw_data.index <= slice_end)]
        sliced_data_stats = pd.DataFrame([sliced_data.mean(), sliced_data.std()], index=['mean','std'])

        if save_all_parameters.value:
            sliced_data.to_csv(filename1)
            sliced_data_stats.to_csv(filename2)
        else:
            sliced_data[slicemap.map.selected].to_csv(filename1)
            sliced_data_stats[slicemap.map.selected].to_csv(filename2)

    # zipping it all
    #rm old zipped file
    zip_filename = filepath+'TP_zipped.zip'
    bashCommand = f"rm {zip_filename}"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    #zip files
    process = subprocess.Popen(['zip',zip_filename] + filename_collector, stdout=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        file_msg = error
    elif not filename_collector:
        file_msg = 'nothing saved, no test points sliced'
    else:
        file_msg = 'files saved to disk'
    
    #delete data files
    process = subprocess.Popen(['rm'] + filename_collector, stdout=subprocess.PIPE)
    output, error = process.communicate()
    
    save_feedback.value = file_msg 
    
    
    
    
    
def lib_on_click_time_slices_db_radio(change, **kwargs):
    
    current_plot_data = kwargs.get('cpt', None)
    analysis_plot = kwargs.get('anplt', None)
    time_slices_db = kwargs.get('tsdb', None)
    poly_order = kwargs.get('pord', None)
    slicemap = kwargs.get('slcmap', None)
    zoom_slider = kwargs.get('zsld', None)
    
    #everytime we select a slice, the analysis plot needs to be updated
    #using the parameters selected on the map and the TS from the dictionary

    if (change['type'] == 'change') and (change['name'] == 'value') and (len(slicemap.map.selected) != 0) and (change['new'] != None):       
        analysis_plot.x_data_slice_min = time_slices_db[change['new']][1]
        analysis_plot.x_data_slice_max = time_slices_db[change['new']][2]
        analysis_plot.update_plot(current_plot_data, slicemap.map.selected, time_slices_db[change['new']][1],
                                  time_slices_db[change['new']][2], poly_order)
        zoom_slider.resetSlider()


        

# slice trim to zoomed figure logic
def slice_trim(**kwargs):
    
    zoom_slider = kwargs.get('zsld', None)
    analysis_plot = kwargs.get('anplt', None)
    time_slices_db = kwargs.get('tsdb', None)
    time_slices_db_radio = kwargs.get('tsdbr', None)
    current_plot_data = kwargs.get('cpt', None)
    poly_order = kwargs.get('pord', None)
    slicemap = kwargs.get('slcmap', None)

    delta_time = np.timedelta64(np.datetime64(analysis_plot.x_data_slice_max, 'us') - np.datetime64(analysis_plot.x_data_slice_min, 'us'))
    #min_delta = np.array((zoom_slider.get_slider_values()[0]/100*delta_time).astype(datetime), dtype="timedelta64[us]")
    #max_delta = np.array((zoom_slider.get_slider_values()[1]/100*delta_time).astype(datetime), dtype="timedelta64[us]")
    min_delta = zoom_slider.get_slider_values()[0] / 100 * delta_time
    max_delta = zoom_slider.get_slider_values()[1] / 100 * delta_time
    current_slice = time_slices_db.get(time_slices_db_radio.value)
    current_slice[2] = current_slice[1] + max_delta
    current_slice[1] = current_slice[1] + min_delta
    time_slices_db[time_slices_db_radio.value] = current_slice
    zoom_slider.resetSlider()
    analysis_plot.x_data_slice_min = current_slice[1]
    analysis_plot.x_data_slice_max = current_slice[2]
    analysis_plot.update_plot(current_plot_data, slicemap.map.selected, 
                              current_slice[1],
                              current_slice[2],
                              poly_order)



def update_analysis_plot(current_plot_data, slicemap_map_selected, plot_object, poly_order, zoom_slider):
    
    delta_time = np.datetime64(plot_object.x_data_slice_max, 'us') - np.datetime64(plot_object.x_data_slice_min, 'us')
    new_xs_min = (np.datetime64(plot_object.x_data_slice_min, 'us') + zoom_slider.get_slider_values()[0]/100*delta_time)
    new_xs_max = (np.datetime64(plot_object.x_data_slice_min, 'us') + zoom_slider.get_slider_values()[1]/100*delta_time)
    
    if new_xs_min < new_xs_max :
        plot_object.update_plot(current_plot_data, slicemap_map_selected, 
                              np.datetime64(new_xs_min, 'us'),
                              np.datetime64(new_xs_max, 'us'),
                              poly_order)

def get_time(t):
    #simple PDAS converter
    #keep it for now
    m = int(t / 60)
    h = int(m / 60)
    s = t - h * 60 * 60 - m * 60
    return (f'{h:02d}:{m:02d}:{s}')

def weeksecondstoutc(gpsweek,gpsseconds, delta_millis, leapseconds):
    '''
    This function credit:
    https://gist.github.com/jeremiahajohnson
    '''
    datetimeformat = "%Y-%m-%d %H:%M:%S"
    datetimeformat_out = "%Y-%m-%d %H:%M:%S.%f"
    epoch = datetime.datetime.strptime("1980-01-06 00:00:00",datetimeformat)
    elapsed = datetime.timedelta(days=(gpsweek*7),seconds=(int(gpsseconds)+leapseconds), milliseconds=delta_millis*1000)
    return datetime.datetime.strftime(epoch + elapsed,datetimeformat_out)


def G3Xweeksecondstoutc(df, leapseconds):
    '''
    This function credit:
    https://gist.github.com/jeremiahajohnson
    '''
    datetimeformat = "%Y-%m-%d %H:%M:%S"
    datetimeformat_out = "%Y-%m-%d %H:%M:%S.%f"
    epoch = datetime.datetime.strptime(df['Local Date'] + " " + "00:00:00", datetimeformat)
    elapsed = datetime.timedelta(seconds=(df['GPS Time of Week'] + leapseconds))
    return datetime.datetime.strftime(epoch + elapsed,datetimeformat_out)



# parameter map class
class ParameterMap:
    def __init__(self, sub_sampled_data, title, map_groups):
        # parameter map and its tool tip
        self.sub_sampled_data = sub_sampled_data
        self.map_names = list(self.sub_sampled_data.columns)
        self.map_codes = [i for i in range(len(self.map_names))]
        #tratar - criar lista de dicionario
        self.map_groups = []
        for key in map_groups.keys():
            self.map_groups.append(map_groups.get(key))
        
        self.map = MarketMap(names=self.map_names, groups=self.map_groups,     
                               layout=Layout(min_width='50px', min_height='70px'),
                                 enable_hover=False, cols=3,
                            map_margin={'top':50, 'bottom':0, 'left':0, 'right':25})

        #self.map.colors = ['#0a141b', '#152837', '#203c53', '#2b506f', '#36648b', '#4f4f4f', '#545454', '#595959', '#5e5e5e']
        self.map.font_style = {'font-size': '10px', 'fill':'white'}
        self.map.title = title
        self.map.title_style = {'fill': 'Red'}
        
    def update_map(self, sub_sampled_data, map_groups):
        self.sub_sampled_data = sub_sampled_data
        self.map_names = list(self.sub_sampled_data.columns)
        self.map_codes = [i for i in range(len(self.map_names))]
        self.map_groups = []
        for key in map_groups.keys():
            self.map_groups.append(map_groups.get(key))
        
        self.map.names = self.map_names
        self.map.groups = self.map_groups


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
        
        self.start_label = widgets.Label('Time slice starting at:    ')
        self.end_label = widgets.Label('..........and finishing at:    ')
        self.boxes_item_layout = Layout(height='', min_width='40px')
        self.slice_start = [self.start_label, self.start_hour_box, self.start_minute_box, self.start_second_box]
        self.slice_end = [self.end_label, self.end_hour_box, self.end_minute_box, self.end_second_box]
        self.boxes_layout = Layout(overflow_x='scroll',
                    border='1px solid black',
                    height='',
                    flex_direction='row',
                    display='flex')
        self.slice_box = VBox([HBox(self.slice_start), HBox(self.slice_end)])
        
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

        self.delta_time_int = np.timedelta64(self.maxval - self.minval)
        self.my_slider_layout = Layout(max_width='100%', width='80%', height='75px')
        
        
        self.LH_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=100,
            step=1,
            description='Left:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            layout=self.my_slider_layout
        )

        self.RH_slider = widgets.IntSlider(
            value=100,
            min=0,
            max=100,
            step=1,
            description='Right:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            layout=self.my_slider_layout
        )
        
    def get_slider_values(self):
        return [self.LH_slider.value, self.RH_slider.value]

    
    def updateScale(self, plot):
        self.minval = np.datetime64(plot.xs.min, 'us') #type datetime.datetime - from bqplot
        self.maxval = np.datetime64(plot.xs.max, 'us')
        plot.x_data_slice_min = self.minval
        plot.x_data_slice_max = self.maxval

        self.delta_time_int = np.timedelta64(self.maxval-self.minval)
        
    def resetSlider(self):
        self.LH_slider.value = 0
        self.RH_slider.value = 100


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
    def update_plot(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        #self.xax.scale.min = x_data.min()
        #self.xax.scale.max = x_data.max()
        



########
class LinePlotBrush(LinePlot):
    def __init__(self, x_data, y_data):
        super().__init__(x_data, y_data)
    
        self.brushintsel = BrushIntervalSelector(scale=self.xs)

        self.fig = Figure(marks=[self.line], axes=[self.xax, self.yax], layout=Layout(width = '80%'), interaction=self.brushintsel)
    
    
    ##########
########

########
class AnalysisPlot(LinePlotBrush):
    def __init__(self, x_data, y_data):
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
        
        self.xs.min=min([mark.x.min() for mark in self.fig.marks]) #this is datetime.datetime internally to bqplot
        self.xs.max=max([mark.x.max() for mark in self.fig.marks])

        self.fit_statistics = widgets.HTML(
                                        value="Empty <b>Empty</b>",
                                        placeholder='Poly Coefs',
                                        description='Poly Coefs',
                                    )
        self.fit_statistics.value = 'Empty'
        
        
    def update_plot(self, current_plot_data, parameter_list, slice_start, slice_end, poly_degree):
        '''
        current_plot_data
        parameter_list
        slice_start: numpy.datetime64, 'us' - from time_slices_db
        slice_end: numpy.datetime64, 'us'
        poly_degrees: int
        '''
        if parameter_list: #this means the list is not empty
            
            self.xs.min = slice_start
            self.xs.max = slice_end
            
            
            slice_start_index = current_plot_data.index.searchsorted(slice_start)
            initial_value = current_plot_data.index[slice_start_index].timestamp()
            xdata = current_plot_data.iloc[(current_plot_data.index >= slice_start) & 
                                           (current_plot_data.index <= slice_end)].index.astype(int).astype(float) \
                                           - (initial_value*1e9)
            xdata = xdata / 1e9  ##
            # ydata is going to be the first selected parameter, for now, thus the "0" below.
            ydata = current_plot_data.iloc[(current_plot_data.index >= slice_start) & 
                                           (current_plot_data.index <= slice_end), 0].values
            if (len(xdata) > poly_degree): #only calculate fit if enough points
                linefit = np.polyfit(xdata, ydata, poly_degree)
                fittedx = np.linspace(xdata[0], xdata[-1], len(xdata))

                fittedy = np.polyval(linefit, fittedx)

                self.fitted_line.x = pd.to_datetime((fittedx*1e9)+(initial_value*1e9))  ##
                self.fitted_line.y = fittedy
                self.ys.min = float(ydata.min(axis=0)) #the scale needs a float...
                self.ys.max = float(ydata.max(axis=0))
                stats_string = 'Highest to lowest expoents: '

                for x in linefit:
                    stats_string = stats_string + '; {:.6f}'.format(x) + '; '
                stats_string = stats_string + '</P> Stats: Min X= {:.2f}'.format(xdata[0]) + ';  '
                stats_string = stats_string + 'Max X= {:.2f}'.format(xdata[-1]) + ';  '
                stats_string = stats_string + 'Avg Y= {:.2f}'.format(ydata.mean()) + ';  '
                stats_string = stats_string + 'Std Y= {:.2f}'.format(ydata.std())
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
            self.msg = 'Test Point not set yet.'
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

####################################################################################################################################
def load_data(unit_test, f, file_status_label):
    # Data Import

    if unit_test:
        # creating fake data just to test
        rng = pd.date_range('27/10/2018 13:00:00', periods=5000, freq='50ms')
        raw_data = pd.DataFrame(data=rng, columns=['Time'])
        raw_data.set_index(['Time'], inplace=True)
        raw_data['counter'] = np.arange(len(raw_data))

        raw_data['fake_angle'] = raw_data.apply(lambda x: (raw_data['counter']/10)%(2*np.pi))
        raw_data['all_zeroes'] = np.zeros(len(raw_data.index.values))
        raw_data['sin0'] = raw_data['counter'] #I have no idea why, but if I do not first create 'sine1' and then calculate with lambda, it throwns me an error
        raw_data['sin0'] = raw_data.apply(lambda x: np.sin(raw_data['fake_angle']))
        raw_data['cos0'] = raw_data['counter']
        raw_data['cos0'] = raw_data.apply(lambda x: np.cos(raw_data['fake_angle']))
        for i in range(5):
            sine_label = 'sin' + str(i+1)
            cosine_label = 'cos' + str(i+1)
            raw_data[sine_label] = raw_data['sin0']
            raw_data[cosine_label] = raw_data['cos0']
        filetype = 'IADS'
    else:

        filename = f.selected
        filetype = 'IADS'
        with open(filename, errors='ignore') as fp:
            first_line = fp.readline()
        if 'iLevil' in first_line:
            file_status_label.value = 'iLevil file detected'
            raw_data=pd.read_csv(filename, encoding='latin1', low_memory=False, skiprows=5)
            raw_data['Time'] = raw_data['UTC_TIME'] + "." + raw_data['TIMER(ms)'].apply(str)
            filetype = 'ilevil'
            file_status_label.value = 'File input finished.'
            raw_data.drop(['LEGEND','DATE', 'UTC_TIME'],axis=1, inplace=True)

        elif 'Analog 1' in first_line:
            file_status_label.value = 'PDAS file detected'
            raw_data=pd.read_csv(filename, encoding='latin1', low_memory=False)
            #raw_data['Time'] = raw_data['Time (s)'].apply(get_time)
            raw_data['delta_seconds'] = pd.to_timedelta(raw_data['Time (s)'] - raw_data['Time (s)'][0], unit='s')
            raw_data['Time'] = pd.to_datetime(weeksecondstoutc(float(raw_data['GNSS Week'][0]),float(raw_data['GNSS TOW (s)'][0]), 0,18)) + raw_data['delta_seconds']
            raw_data.drop(['delta_seconds'], axis=1, inplace=True)
            filetype = 'PDAS'
            file_status_label.value = 'File input finished.'

        elif '#airframe_info' in first_line:
            file_status_label.value = 'G3X file detected'
            raw_data=pd.read_csv(filename, encoding='latin1', low_memory=False, skiprows=[0,2,3,4,5,6,7,8,9,10]) #this is necessary to clean empty rows at the start of the file
            raw_data['Time'] = raw_data.apply(G3Xweeksecondstoutc, args= (-18,), axis=1)
            raw_data.drop(['Date (yyyy-mm-dd)', 'Time (hh:mm:ss)', 'UTC Time (hh:mm:ss)', 'UTC Offset (hh:mm)', ], axis=1, inplace=True)
            filetype = 'G3X'
            file_status_label.value = 'File input finished.'


        elif len(first_line) == 1:
            file_status_label.value = 'X-Plane file detected'
            raw_data=pd.read_csv(filename, encoding='latin1', low_memory=False, skiprows=1, delimiter='|')
            raw_data['Time'] = raw_data['   _real,_time '].apply(get_time)
            filetype = 'X-Plane'
            file_status_label.value = 'File input finished.'

        else:
            file_status_label.value = 'Reading IADS file'
            raw_data=pd.read_csv(filename, encoding='latin1', low_memory=False)
            file_status_label.value = 'File input finished.'
            dirty_file = False
            if raw_data['Time'][10].count(':') > 2:
                dirty_file = True

            if dirty_file == True:
                ## FOR DIRTY DATA
                file_status_label.value = 'Dirty file detected ... cleaning up the data...'  #DIRTY DATA
                raw_data.fillna(value=0, inplace=True)  #DIRTY DATA
                raw_data['Time'] = (raw_data['Time'].str.slice_replace(0,4,''))  #DIRTY DATA

        raw_data['Time'] = pd.to_datetime(raw_data['Time'])   #CLEAN DATA
        raw_data = raw_data.set_index(['Time'])   #CLEAN DATA
    return raw_data, filetype


        
#####################################################################################################################################
# PARAMETER GROUPS MAPS
#
# Add for each aircraft

def get_param_group(filepath, filetype, raw_data, in_map_groups):
    out_map_groups = in_map_groups
    if 'FPQA' in filepath:
        FPQA_dict = {}       
        FPQA_dict['ALPHA_EU'] = 'anemo'
        #FPQA_dict['AmbDensRatio_Aero'] = ''
        #FPQA_dict['AmbPressRatio_Aero'] = ''
        #FPQA_dict['AmbTempRatio_Aero'] = ''
        FPQA_dict['BETA_EU'] = 'anemo'
        FPQA_dict['COLLPer_EU'] = 'commands'
        #FPQA_dict['Delta_Wref'] = ''
        FPQA_dict['DENSITY_ALT'] = 'anemo'
        FPQA_dict['EAS_Aero'] = 'anemo'
        FPQA_dict['ENG_1_TQ_PER'] = 'engine'
        FPQA_dict['ENG_2_TQ_PER'] = 'engine'
        FPQA_dict['Eng1_FF_17'] = 'engine'
        FPQA_dict['Eng1_FF_18'] = 'engine'
        FPQA_dict['Eng1_FF_19'] = 'engine'
        FPQA_dict['Eng1_FF_20'] = 'engine'
        FPQA_dict['Eng1_FF_21'] = 'engine'
        FPQA_dict['Eng1_FF_22'] = 'engine'
        FPQA_dict['Eng1_FU_38'] = 'engine'
        FPQA_dict['Eng1_FU_39'] = 'engine'
        FPQA_dict['Eng1_FU_40'] = 'engine'
        FPQA_dict['Eng1_FU_41'] = 'engine'
        FPQA_dict['Eng1_FU_42'] = 'engine'
        FPQA_dict['Eng1_FU_43'] = 'engine'
        FPQA_dict['Eng1_FuelF_EU'] = 'engine'
        FPQA_dict['Eng1_FuelU_EU'] = 'engine'
        FPQA_dict['Eng2_FF_24'] = 'engine'
        FPQA_dict['Eng2_FF_25'] = 'engine'
        FPQA_dict['Eng2_FF_26'] = 'engine'
        FPQA_dict['Eng2_FF_27'] = 'engine'
        FPQA_dict['Eng2_FF_28'] = 'engine'
        FPQA_dict['Eng2_FF_29'] = 'engine'
        FPQA_dict['Eng2_FU_45'] = 'engine'
        FPQA_dict['Eng2_FU_46'] = 'engine'
        FPQA_dict['Eng2_FU_47'] = 'engine'
        FPQA_dict['Eng2_FU_48'] = 'engine'
        FPQA_dict['Eng2_FU_49'] = 'engine'
        FPQA_dict['Eng2_FU_50'] = 'engine'
        FPQA_dict['Eng2_FuelF_EU'] = 'engine'
        FPQA_dict['Eng2_FuelU_EU'] = 'engine'
        #FPQA_dict['Event_EU'] = ''
        FPQA_dict['Fuel_left_kg'] = 'engine'
        FPQA_dict['Fuel_R3'] = 'engine'
        FPQA_dict['Fuel_R4'] = 'engine'
        FPQA_dict['Fuel_R5'] = 'engine'
        FPQA_dict['Fuel_R6'] = 'engine'
        FPQA_dict['Fuel_R7'] = 'engine'
        FPQA_dict['Fuel_R8'] = 'engine'
        FPQA_dict['Fuel_Units'] = 'engine'
        FPQA_dict['G_Alt'] = 'INS'
        FPQA_dict['G_GS'] = 'INS'
        FPQA_dict['G_LatDeg'] = 'INS'
        FPQA_dict['G_LatMinDD'] = 'INS'
        FPQA_dict['G_LatSecDD'] = 'INS'
        FPQA_dict['G_LongDeg'] = 'INS'
        FPQA_dict['G_LongMinDD'] = 'INS'
        FPQA_dict['G_LongSecDD'] = 'INS'
        FPQA_dict['G_Track'] = 'INS'
        FPQA_dict['Garmin_EW'] = 'INS'
        FPQA_dict['Garmin_LatDD'] = 'INS'
        FPQA_dict['Garmin_LatSign'] = 'INS'
        FPQA_dict['Garmin_LongDD'] = 'INS'
        FPQA_dict['Garmin_LongSign'] = 'INS'
        FPQA_dict['Garmin_NS'] = 'INS'
        FPQA_dict['Hp_bottom'] = 'anemo'
        FPQA_dict['Hp_top'] = 'anemo'
        FPQA_dict['IAS_Aero'] = 'anemo'
        FPQA_dict['ISA_Press'] = 'anemo'
        FPQA_dict['ISA_TempC'] = 'anemo'
        FPQA_dict['ISA_TempK'] = 'anemo'
        FPQA_dict['LATPer_EU'] = 'commands'
        FPQA_dict['LONGPer_EU'] = 'commands'
        FPQA_dict['MACH_Aero'] = 'anemo'
        FPQA_dict['N1_ENG_1_SEC'] = 'engine'
        FPQA_dict['N1_ENG_2_SEC'] = 'engine'
        FPQA_dict['N2_ENG_1_SEC'] = 'engine'
        FPQA_dict['N2_ENG_2_SEC'] = 'engine'
        FPQA_dict['NF_1_PER'] = 'engine'
        FPQA_dict['NF_2_PER'] = 'engine'
        FPQA_dict['NG_1_PER'] = 'engine'
        FPQA_dict['NG_2_PER'] = 'engine'
        FPQA_dict['NR_ROT_PER'] = 'rotor'
        #FPQA_dict['NR_ROT_SEC'] = ''
        FPQA_dict['OAT'] = 'anemo'
        FPQA_dict['OAT_EU'] = 'anemo'
        FPQA_dict['OIL_TEMP_ENG_1_EU'] = 'engine'
        FPQA_dict['OIL_TEMP_ENG_2_EU'] = 'engine'
        FPQA_dict['PITCH_ACC'] = 'INS'
        FPQA_dict['PITOT_EU'] = 'anemo'
        FPQA_dict['PRESS_ALT'] = 'anemo'
        FPQA_dict['ROLL_ACC'] = 'INS'
        FPQA_dict['RUDDERPer_EU'] = 'commands'
        FPQA_dict['SBG_Accel_X'] = 'INS'
        FPQA_dict['SBG_Accel_Y'] = 'INS'
        FPQA_dict['SBG_Accel_Z'] = 'INS'
        FPQA_dict['SBG_Alt'] = 'INS'
        FPQA_dict['SBG_AltHiB'] = 'INS'
        FPQA_dict['SBG_AltLowB'] = 'INS'
        FPQA_dict['SBG_Day'] = 'INS'
        FPQA_dict['SBG_DN_Vel'] = 'INS'
        FPQA_dict['SBG_E_Vel'] = 'INS'
        FPQA_dict['SBG_GS'] = 'INS'
        FPQA_dict['SBG_Gyro_X'] = 'INS'
        FPQA_dict['SBG_Gyro_Y'] = 'INS'
        FPQA_dict['SBG_Gyro_Z'] = 'INS'
        FPQA_dict['SBG_Hr'] = 'INS'
        FPQA_dict['SBG_Lat'] = 'INS'
        FPQA_dict['SBG_LatHiB'] = 'INS'
        FPQA_dict['SBG_LatLowB'] = 'INS'
        FPQA_dict['SBG_Long'] = 'INS'
        FPQA_dict['SBG_LongHiB'] = 'INS'
        FPQA_dict['SBG_LongLowB'] = 'INS'
        FPQA_dict['SBG_Min'] = 'INS'
        FPQA_dict['SBG_Mth'] = 'INS'
        FPQA_dict['SBG_N_Vel'] = 'INS'
        FPQA_dict['SBG_NanoSec'] = 'INS'
        FPQA_dict['SBG_NanoSec_EU'] = 'INS'
        FPQA_dict['SBG_Nx_g'] = 'INS'
        FPQA_dict['SBG_Ny_g'] = 'INS'
        FPQA_dict['SBG_Nz_g'] = 'INS'
        FPQA_dict['SBG_Pitch'] = 'INS'
        FPQA_dict['SBG_PitchAngle'] = 'INS'
        FPQA_dict['SBG_PitchRate'] = 'INS'
        FPQA_dict['SBG_Roll'] = 'INS'
        FPQA_dict['SBG_RollAngle'] = 'INS'
        FPQA_dict['SBG_RollRate'] = 'INS'
        FPQA_dict['SBG_Sec'] = 'INS'
        FPQA_dict['SBG_Track'] = 'INS'
        FPQA_dict['SBG_Yaw'] = 'INS'
        FPQA_dict['SBG_YawAngle'] = 'INS'
        FPQA_dict['SBG_YawRate'] = 'INS'
        FPQA_dict['SBG_Yr'] = 'INS'
        FPQA_dict['STATIC_EU'] = 'anemo'
        FPQA_dict['Target_Hp'] = 'anemo'
        FPQA_dict['Target_Wref'] = ''
        FPQA_dict['TAS_Aero'] = 'anemo'
        FPQA_dict['TOT_1'] = 'engine'
        FPQA_dict['TOT_1_EU'] = 'engine'
        FPQA_dict['TOT_2'] = 'engine'
        FPQA_dict['TOT_2_EU'] = 'engine'
        FPQA_dict['TotFuel_Rmng_EU'] = 'engine'
        FPQA_dict['TotFuelF_10'] = 'engine'
        FPQA_dict['TotFuelF_11'] = 'engine'
        FPQA_dict['TotFuelF_12'] = 'engine'
        FPQA_dict['TotFuelF_13'] = 'engine'
        FPQA_dict['TotFuelF_14'] = 'engine'
        FPQA_dict['TotFuelF_15'] = 'engine'
        FPQA_dict['TotFuelFlow_EU'] = 'engine'
        FPQA_dict['TQ'] = 'engine'
        FPQA_dict['TQ_ref'] = 'engine'
        FPQA_dict['Wref'] = ''
        FPQA_dict['Wref100'] = ''
        FPQA_dict['Wref102'] = ''
        FPQA_dict['Wref102'] = ''
        FPQA_dict['Wref98'] = ''
        FPQA_dict['Wref98'] = ''
        FPQA_dict['YAW_ACC'] = 'INS'
        FPQA_dict['ZFW'] = ''
        FPQA_dict['VN_NZ'] = 'INS'
        FPQA_dict['YAW_A'] = 'INS'
        FPQA_dict['YAW_R'] = 'INS'
        FPQA_dict['PITCH ANGLE'] = 'INS'
        FPQA_dict['PITCH RATE'] = 'INS'
        FPQA_dict['ROLL ANGLE'] = 'INS'
        FPQA_dict['ROLL RATE'] = 'INS'
        FPQA_dict['UTC YEAR'] = 'INS'
        FPQA_dict['UTC MONTH'] = 'INS'
        FPQA_dict['UTC DAY'] = 'INS'
        FPQA_dict['UTC HOUR'] = 'INS'
        FPQA_dict['UTC MINUTE'] = 'INS'
        FPQA_dict['UTC SECOND'] = 'INS'
        FPQA_dict['UTC MILI SECOND'] = 'INS'
        FPQA_dict['VECTORNAV ALTITUDE'] = 'INS'
        FPQA_dict['VECTORNAV DOWN VELOCITY'] = 'INS'
        FPQA_dict['VECTORNAV EAST VELOCITY'] = 'INS'
        FPQA_dict['VECTORNAV LATITUDE'] = 'INS'
        FPQA_dict['VECTORNAV LONGITUDE'] = 'INS'
        FPQA_dict['VECTORNAV NORTH VELOCITY'] = 'INS'
        FPQA_dict['VECTORNAV ACCELERATION X'] = 'INS'
        FPQA_dict['VECTORNAV ACCELERATION Y'] = 'INS'
        FPQA_dict['VECTORNAV ACCELERATION Z'] = 'INS'
        FPQA_dict['YAW ANGLE'] = 'INS'
        FPQA_dict['YAW RATE'] = 'INS'


        for key in in_map_groups.keys():
            if FPQA_dict.get(key):
                out_map_groups[key] = FPQA_dict[key]
                
    elif 'FXZI' in filepath:
        FXZI_dict = {}

        FXZI_dict['Wref100'] = 'FTI'
        FXZI_dict['Wref102'] = 'FTI'
        FXZI_dict['Wref98'] = 'FTI'
        FXZI_dict['Wref'] = 'FTI'
        
        FXZI_dict['AIRSPEED_PASCAL'] = 'anemo'
        FXZI_dict['AIRSPEED_Pd'] = 'anemo'
        FXZI_dict['AIRSPEED'] = 'anemo'
        FXZI_dict['ALPHA_EU'] = 'anemo'
        FXZI_dict['ALTITUDE_FEET'] = 'anemo'
        FXZI_dict['ALTITUDE_PASCAL'] = 'anemo'
        FXZI_dict['BETA_EU'] = 'anemo'
        FXZI_dict['DIVE_EU'] = 'commands'
        FXZI_dict['EGT_EU_1'] = 'engine'
        FXZI_dict['ELEV_TRIM_POS_EU'] = 'commands'
        FXZI_dict['Eng1_FuelF_EU'] = 'engine'
        FXZI_dict['Eng1_FuelU_EU'] = 'engine'
        FXZI_dict['Eng2_FuelF_EU'] = 'engine'
        FXZI_dict['Eng2_FuelU_EU'] = 'engine'
        FXZI_dict['EOP_EU'] = 'engine'
        FXZI_dict['EOT_EU'] = 'engine'
        FXZI_dict['ERPM_EU'] = 'engine'
        FXZI_dict['EVENT_1_EU'] = 'FTI'
        FXZI_dict['EVENT_2_EU'] = 'FTI'
        FXZI_dict['FLAP_POS_EU'] = 'commands'
        FXZI_dict['Fuel_left_kg'] = 'engine'
        FXZI_dict['G_Alt'] = 'Garmin'
        FXZI_dict['G_GS'] = 'Garmin'
        FXZI_dict['G_Track'] = 'Garmin'
        FXZI_dict['Garmin_LatDD'] = 'Garmin'
        FXZI_dict['Garmin_LongDD'] = 'Garmin'
        FXZI_dict['L_RUD_FORCE_EU'] = 'commands'
        FXZI_dict['LAT_CTL_POS_EU'] = 'commands'
        FXZI_dict['LAT_FORCE_EU'] = 'commands'
        FXZI_dict['LG_STS_EU'] = 'INS'
        FXZI_dict['LONG_CTL_POS_EU'] = 'commands'
        FXZI_dict['LONG_FORCE_EU'] = 'commands'
        FXZI_dict['NxC_EU'] = 'FTI'
        FXZI_dict['NxL_EU'] = 'FTI'
        FXZI_dict['NxR_EU'] = 'FTI'
        FXZI_dict['NyC_EU'] = 'FTI'
        FXZI_dict['NyL_EU'] = 'FTI'
        FXZI_dict['NyR_EU'] = 'FTI'
        FXZI_dict['NzC_EU'] = 'FTI'
        FXZI_dict['NzL_EU'] = 'FTI'
        FXZI_dict['NzR_EU'] = 'FTI'
        FXZI_dict['OAT_EU'] = 'anemo'
        FXZI_dict['R_RUD_FORCE_EU'] = 'commands'
        FXZI_dict['RUD_PDL_POS_EU'] = 'commands'
        FXZI_dict['THROT_POS_EU'] = 'commands'
        FXZI_dict['TotFuel_Rmng_EU'] = 'engine'
        FXZI_dict['TotFuelFlow_EU'] = 'engine'
        FXZI_dict['VN_Alt'] = 'INS'
        FXZI_dict['VN_DnVel'] = 'INS'
        FXZI_dict['VN_DTheta0'] = 'INS'
        FXZI_dict['VN_DTheta1'] = 'INS'
        FXZI_dict['VN_DTheta2'] = 'INS'
        FXZI_dict['VN_Evel'] = 'INS'
        FXZI_dict['VN_GPSTime'] = 'INS'
        FXZI_dict['VN_Lat'] = 'INS'
        FXZI_dict['VN_Long'] = 'INS'
        FXZI_dict['VN_Mag0'] = 'INS'
        FXZI_dict['VN_Mag1'] = 'INS'
        FXZI_dict['VN_Mag2'] = 'INS'
        FXZI_dict['VN_NVel'] = 'INS'
        FXZI_dict['VN_Nx'] = 'INS'
        FXZI_dict['VN_Ny'] = 'INS'
        FXZI_dict['VN_Nz'] = 'INS'
        FXZI_dict['VN_PitchAng'] = 'INS'
        FXZI_dict['VN_PitchRate'] = 'INS'
        FXZI_dict['VN_PosU'] = 'INS'
        FXZI_dict['VN_Press'] = 'INS'
        FXZI_dict['VN_RollAng'] = 'INS'
        FXZI_dict['VN_RollRate'] = 'INS'
        FXZI_dict['VN_Temp'] = 'INS'
        FXZI_dict['VN_UnCompNx'] = 'INS'
        FXZI_dict['VN_UnCompNy'] = 'INS'
        FXZI_dict['VN_UnCompNz'] = 'INS'
        FXZI_dict['VN_UnCompPitchRate'] = 'INS'
        FXZI_dict['VN_UnCompRollRate'] = 'INS'
        FXZI_dict['VN_UnCompYawRate'] = 'INS'
        FXZI_dict['VN_VelU'] = 'INS'
        FXZI_dict['VN_YawAng'] = 'INS'
        FXZI_dict['VN_YawRate'] = 'INS'
        
        for key in in_map_groups.keys():
            if FXZI_dict.get(key):
                out_map_groups[key] = FXZI_dict[key]
        
    elif 'N7025J' in filepath:
        N7025J_dict = {}
        N7025J_dict['Ail_POS_Cmd_Perc'] = 'commands'
        N7025J_dict['AIL_POS_Command'] = 'commands'
        N7025J_dict['AIL_POS_Surface_LH'] = 'commands'
        N7025J_dict['AIL_POS_Surface_RH'] = 'commands'
        N7025J_dict['Elev_POS_Cmd_Perc'] = 'commands'
        N7025J_dict['ELEV_POS_Command'] = 'commands'
        N7025J_dict['ELEV_POS_EU'] = 'commands'
        N7025J_dict['EVENT_EU'] = 'INS'
        N7025J_dict['L_RUDPEDF'] = 'commands'
        N7025J_dict['LATF'] = 'commands'
        N7025J_dict['LONGF'] = 'commands'
        N7025J_dict['OAT_EU'] = 'anemo'
        N7025J_dict['R_RUDPEDF'] = 'commands'
        N7025J_dict['Rud_Diff_Force'] = 'commands'
        N7025J_dict['Rud_POS_Cmd_Perc'] = 'commands'
        N7025J_dict['RUD_POS_Command'] = 'commands'
        N7025J_dict['RUD_Surface_POS_EU'] = 'commands'
        N7025J_dict['VN_Alt'] = 'INS'
        N7025J_dict['VN_DnVel'] = 'INS'
        N7025J_dict['VN_Evel'] = 'INS'
        N7025J_dict['VN_Lat'] = 'INS'
        N7025J_dict['VN_Long'] = 'INS'
        N7025J_dict['VN_NVel'] = 'INS'
        N7025J_dict['VN_Nx'] = 'INS'
        N7025J_dict['VN_Ny'] = 'INS'
        N7025J_dict['VN_Nz'] = 'INS'
        N7025J_dict['VN_PitchAng'] = 'INS'
        N7025J_dict['VN_PitchRate'] = 'INS'
        N7025J_dict['VN_Press'] = 'INS'
        N7025J_dict['VN_RollAng'] = 'INS'
        N7025J_dict['VN_RollRate'] = 'INS'
        N7025J_dict['VN_Temp'] = 'INS'
        N7025J_dict['VN_YawAng'] = 'INS'
        N7025J_dict['VN_YawRate'] = 'INS'
        
        for key in in_map_groups.keys():
            if N7025J_dict.get(key):
                out_map_groups[key] = N7025J_dict[key]
                
    elif 'PDAS' in filetype:
        PDAS_dict = {}      
        #PDAS_dict['Time (s)'] = 'INS'
        #PDAS_dict['Active'] = 'INS'
        #PDAS_dict['Test Point'] = 'INS'
        PDAS_dict['IMU Accel X (g)'] = 'INS'
        PDAS_dict['IMU Accel Y (g)'] = 'INS'
        PDAS_dict['IMU Accel Z (g)'] = 'INS'
        PDAS_dict['IMU Gyro X (deg/s)'] = 'INS'
        PDAS_dict['IMU Gyro Y (deg/s)'] = 'INS'
        PDAS_dict['IMU Gyro Z (deg/s)'] = 'INS'
        PDAS_dict['IMU Mag X (uT)'] = 'INS'
        PDAS_dict['IMU Mag Y (uT)'] = 'INS'
        PDAS_dict['IMU Mag Z (uT)'] = 'INS'
        PDAS_dict['GNSS TOW (s)'] = 'INS'
        PDAS_dict['GNSS Week'] = 'INS'
        PDAS_dict['GNSS Num Sat'] = 'INS'
        PDAS_dict['GNSS Fix'] = 'INS'
        PDAS_dict['GNSS Latitude (deg)'] = 'INS'
        PDAS_dict['GNSS Longitude (deg)'] = 'INS'
        PDAS_dict['GNSS Altitude (ft)'] = 'INS'
        PDAS_dict['GNSS North Velocity (kts)'] = 'INS'
        PDAS_dict['GNSS East Velocity (kts)'] = 'INS'
        PDAS_dict['GNSS Down Velocity (kts)'] = 'INS'
        PDAS_dict['GNSS Horizontal Accuracy (ft)'] = 'INS'
        PDAS_dict['GNSS Vertical Accuracy (ft)'] = 'INS'
        PDAS_dict['GNSS Speed Accuracy (kts)'] = 'INS'
        PDAS_dict['INS Accel X (g)'] = 'INS'
        PDAS_dict['INS Accel Y (g)'] = 'INS'
        PDAS_dict['INS Accel Z (g)'] = 'INS'
        PDAS_dict['INS Gyro X (deg/s)'] = 'INS'
        PDAS_dict['INS Gyro Y (deg/s)'] = 'INS'
        PDAS_dict['INS Gyro Z (deg/s)'] = 'INS'
        PDAS_dict['INS Mag X (uT)'] = 'INS'
        PDAS_dict['INS Mag Y (uT)'] = 'INS'
        PDAS_dict['INS Mag Z (uT)'] = 'INS'
        PDAS_dict['INS Latitude (deg)'] = 'INS'
        PDAS_dict['INS Longitude (deg)'] = 'INS'
        PDAS_dict['INS Altitude (ft)'] = 'INS'
        PDAS_dict['INS North Velocity (kts)'] = 'INS'
        PDAS_dict['INS East Velocity (kts)'] = 'INS'
        PDAS_dict['INS Down Velocity (kts)'] = 'INS'
        PDAS_dict['INS Yaw (deg)'] = 'INS'
        PDAS_dict['INS Pitch (deg)'] = 'INS'
        PDAS_dict['INS Roll (deg)'] = 'INS'
        PDAS_dict['INS Attitude Uncertainty (deg)'] = 'INS'
        PDAS_dict['INS Position Uncertainty (ft)'] = 'INS'
        PDAS_dict['INS Velocity Uncertainty (kts)'] = 'INS'
        PDAS_dict['Pitot-Static Enabled'] = 'anemo'
        PDAS_dict['OAT Enabled'] = 'anemo'
        PDAS_dict['Static Pressure (Pa)'] = 'anemo'
        PDAS_dict['Filtered Static Pressure (Pa)'] = 'anemo'
        PDAS_dict['Differential Pressure (Pa)'] = 'anemo'
        PDAS_dict['Filtered Differential Pressure (Pa)'] = 'anemo'
        PDAS_dict['OAT (C)'] = 'anemo'
        PDAS_dict['Filtered OAT (C)'] = 'anemo'
        PDAS_dict['IAS (kts)'] = 'anemo'
        PDAS_dict['EAS (kts)'] = 'anemo'
        PDAS_dict['TAS (kts)'] = 'anemo'
        PDAS_dict['Pressure Altitude (ft)'] = 'anemo'
        PDAS_dict['Altitude AGL (ft)'] = 'anemo'
        PDAS_dict['Density Altitude (ft)'] = 'anemo'
        PDAS_dict['Analog 0 Enabled'] = 'analog'
        PDAS_dict['Analog 0 Voltage'] = 'analog'
        PDAS_dict['Analog 0 Filtered Voltage'] = 'analog'
        PDAS_dict['Analog 0 Calibrated Value'] = 'analog'
        PDAS_dict['Analog 1 Enabled'] = 'analog'
        PDAS_dict['Analog 1 Voltage'] = 'analog'
        PDAS_dict['Analog 1 Filtered Voltage'] = 'analog'
        PDAS_dict['Analog 1 Calibrated Value'] = 'analog'
        PDAS_dict['Analog 2 Enabled'] = 'analog'
        PDAS_dict['Analog 2 Voltage'] = 'analog'
        PDAS_dict['Analog 2 Filtered Voltage'] = 'analog'
        PDAS_dict['Analog 2 Calibrated Value'] = 'analog'
        PDAS_dict['Analog 3 Enabled'] = 'analog'
        PDAS_dict['Analog 3 Voltage'] = 'analog'
        PDAS_dict['Analog 3 Filtered Voltage'] = 'analog'
        PDAS_dict['Analog 3 Calibrated Value'] = 'analog'
        PDAS_dict['Analog 4 Enabled'] = 'analog'
        PDAS_dict['Analog 4 Voltage'] = 'analog'
        PDAS_dict['Analog 4 Filtered Voltage'] = 'analog'
        PDAS_dict['Analog 4 Calibrated Value'] = 'analog'
        PDAS_dict['Analog 5 Enabled'] = 'analog'
        PDAS_dict['Analog 5 Voltage'] = 'analog'
        PDAS_dict['Analog 5 Filtered Voltage'] = 'analog'
        PDAS_dict['Analog 5 Calibrated Value'] = 'analog'
        PDAS_dict['Analog 6 Enabled'] = 'analog'
        PDAS_dict['Analog 6 Voltage'] = 'analog'
        PDAS_dict['Analog 6 Filtered Voltage'] = 'analog'
        PDAS_dict['Analog 6 Calibrated Value'] = 'analog'
        PDAS_dict['Analog 7 Enabled'] = 'analog'
        PDAS_dict['Analog 7 Voltage'] = 'analog'
        PDAS_dict['Analog 7 Filtered Voltage'] = 'analog'
        PDAS_dict['Analog 7 Calibrated Value'] = 'analog'
        PDAS_dict['Analog 8 Enabled'] = 'analog'
        PDAS_dict['Analog 8 Voltage'] = 'analog'
        PDAS_dict['Analog 8 Filtered Voltage'] = 'analog'
        PDAS_dict['Analog 8 Calibrated Value'] = 'analog'
        PDAS_dict['Analog 9 Enabled'] = 'analog'
        PDAS_dict['Analog 9 Voltage'] = 'analog'
        PDAS_dict['Analog 9 Filtered Voltage'] = 'analog'
        PDAS_dict['Analog 9 Calibrated Value'] = 'analog'
        PDAS_dict['Analog 10 Enabled'] = 'analog'
        PDAS_dict['Analog 10 Voltage'] = 'analog'
        PDAS_dict['Analog 10 Filtered Voltage'] = 'analog'
        PDAS_dict['Analog 10 Calibrated Value'] = 'analog'
        PDAS_dict['Analog 11 Enabled'] = 'analog'
        PDAS_dict['Analog 11 Voltage'] = 'analog'
        PDAS_dict['Analog 11 Filtered Voltage'] = 'analog'
        PDAS_dict['Analog 11 Calibrated Value'] = 'analog'
        #PDAS_dict['delta_seconds'] = 'INS'
        
                
        for key in in_map_groups.keys():
            if PDAS_dict.get(key):
                out_map_groups[key] = PDAS_dict[key]
    

    elif 'ilevil' in filetype:
        ILEVIL_dict = {}       
        #ILEVIL_dict['TIMER(ms)'] = 'INS'
        ILEVIL_dict['LATITUDE'] = 'INS'
        ILEVIL_dict['LONGITUDE'] = 'INS'
        ILEVIL_dict['GNDSPD'] = 'INS'
        ILEVIL_dict['COURSE'] = 'INS'
        ILEVIL_dict['MAG_DECLINATION'] = 'INS'
        ILEVIL_dict['GPS_CLIMBRATE'] = 'INS'
        ILEVIL_dict['ALTGPS'] = 'INS'
        ILEVIL_dict['PALT'] = 'anemo'
        ILEVIL_dict['AIRSPEED'] = 'anemo'
        ILEVIL_dict['PRESSURE_CLIMBRATE'] = 'anemo'
        ILEVIL_dict['ROLL'] = 'INS'
        ILEVIL_dict['PITCH'] = 'INS'
        ILEVIL_dict['YAW'] = 'INS'
        ILEVIL_dict['HEADING'] = 'INS'
        ILEVIL_dict['SLIP'] = 'INS'
        ILEVIL_dict['WAAS'] = 'INS'
        ILEVIL_dict['GPS_FIX'] = 'INS'
        ILEVIL_dict['SATELLITES'] = 'INS'
        ILEVIL_dict['AHRS_STATUS'] = 'INS'
        #ILEVIL_dict['EVENT'] = 'INS'
        ILEVIL_dict['ACC_X'] = 'INS'
        ILEVIL_dict['ACC_Y'] = 'INS'
        ILEVIL_dict['ACC_Z'] = 'INS'
        ILEVIL_dict['RATE_P'] = 'INS'
        ILEVIL_dict['RATE_Q'] = 'INS'
        ILEVIL_dict['RATE_R'] = 'INS'
        ILEVIL_dict['Vx'] = 'INS'
        ILEVIL_dict['Vy'] = 'INS'
        ILEVIL_dict['Vz'] = 'INS'
        #ILEVIL_dict['TEMPERATURE'] = 'INS'
        
        for key in in_map_groups.keys():
            if ILEVIL_dict.get(key):
                out_map_groups[key] = ILEVIL_dict[key]
                

    elif 'G3X' in filetype:
        G3X_dict = {}       
        #ILEVIL_dict['TIMER(ms)'] = 'INS'
        G3X_dict['Local Date'] = 'time'
        G3X_dict['Local Time'] = 'time'
        G3X_dict['UTC Time'] = 'time'
        G3X_dict['UTC Offset'] = 'time'
        G3X_dict['Latitude'] = 'INS'
        G3X_dict['Longitude'] = 'INS'
        G3X_dict['AltGPS'] = 'INS'
        G3X_dict['GPS Fix Status'] = 'INS'
        G3X_dict['GPS Time of Week'] = 'INS'
        G3X_dict['GndSpd'] = 'INS'
        G3X_dict['TRK'] = 'INS'
        G3X_dict['GPS Velocity E'] = 'INS'
        G3X_dict['GPS Velocity N'] = 'INS'
        G3X_dict['GPS Velocity U'] = 'INS'
        G3X_dict['HDG'] = 'INS'
        G3X_dict['GPS PDOP'] = 'INS'
        G3X_dict['GPS Sats'] = 'INS'
        G3X_dict['AltMSL'] = 'anemo'
        G3X_dict['AltB'] = 'anemo'
        G3X_dict['VSpd'] = 'anemo'
        G3X_dict['IAS'] = 'anemo'
        G3X_dict['TAS'] = 'anemo'
        G3X_dict['Pitch'] = 'INS'
        G3X_dict['Roll'] = 'INS'
        G3X_dict['Lateral Acceleration'] = 'INS'
        G3X_dict['Normal Acceleration'] = 'INS'
        G3X_dict['Selected Heading'] = 'NAV'
        G3X_dict['Selected Altitude'] = 'NAV'
        G3X_dict['Baro Setting'] = 'anemo'
        G3X_dict['Active Nav Source'] = 'NAV'
        G3X_dict['Nav Identifier'] = 'NAV'
        G3X_dict['Nav Frequency'] = 'NAV'
        G3X_dict['Nav Distance'] = 'NAV'
        G3X_dict['Nav Bearing'] = 'NAV'
        G3X_dict['Nav Course'] = 'NAV'
        G3X_dict['Cross Track Distance'] = 'NAV'
        G3X_dict['Horizontal CDI Deflection'] = 'NAV'
        G3X_dict['Horizontal CDI Full Scale'] = 'NAV'
        G3X_dict['Horizontal CDI Scale'] = 'NAV'
        G3X_dict['Vertical CDI Deflection'] = 'NAV'
        G3X_dict['Vertical CDI Full Scale'] = 'NAV'
        G3X_dict['VNAV CDI Deflection'] = 'NAV'
        G3X_dict['VNAV Target Altitude'] = 'NAV'
        G3X_dict['Roll Steering'] = 'NAV'
        G3X_dict['MagVar'] = 'NAV'
        G3X_dict['OAT'] = 'anemo'
        G3X_dict['Density Altitude'] = 'anemo'
        G3X_dict['Wind Speed'] = 'INS'
        G3X_dict['Wind Direction'] = 'INS'
        G3X_dict['AHRS Status'] = 'INS'
        G3X_dict['AHRS Dev'] = 'INS'
        G3X_dict['Magnetometer Status'] = 'NAV'
        G3X_dict['Transponder Code'] = 'NAV'
        G3X_dict['Transponder Mode'] = 'NAV'
        G3X_dict['E1 OilT'] = 'engine'
        G3X_dict['Fuel Qty'] = 'engine'
        G3X_dict['E1 FPres'] = 'engine'
        G3X_dict['E1 OilP'] = 'engine'
        G3X_dict['E1 RPM'] = 'engine'
        G3X_dict['E1 Volt1'] = 'engine'
        G3X_dict['E1 Volt2'] = 'engine'
        G3X_dict['E1 Amp1'] = 'engine'
        G3X_dict['E1 FFlow'] = 'engine'
        G3X_dict['E1 RPM2'] = 'engine'
        G3X_dict['Hyd Main'] = 'engine'
        G3X_dict['Hyd Emrg'] = 'engine'
        G3X_dict['E1 EGT1'] = 'engine'
        G3X_dict['CAS Alert'] = 'NAV'
        G3X_dict['Terrain Alert'] = 'NAV'
        #G3X_dict['Event Marker'] = 'INS'
        
        for key in in_map_groups.keys():
            if G3X_dict.get(key):
                out_map_groups[key] = G3X_dict[key]
        
    return out_map_groups