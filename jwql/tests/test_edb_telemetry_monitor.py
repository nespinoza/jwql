#! /usr/bin/env python

"""Tests for the EDB telemetry

Authors
-------

    - Bryan Hilbert

Use
---

    These tests can be run via the command line (omit the ``-s`` to
    suppress verbose output to stdout):
    ::

        pytest -s test_edb_telemetry_monitor.py
"""
from collections import defaultdict
import numpy as np
import os
import pytest

from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.table.column import Column
from astropy.time import Time, TimeDelta
import astropy.units as u
import datetime
import numpy as np

from jwql.edb.engineering_database import EdbMnemonic
from jwql.instrument_monitors.common_monitors import edb_telemetry_monitor as etm
from jwql.instrument_monitors.common_monitors.edb_telemetry_monitor_utils import condition as cond
from jwql.instrument_monitors.common_monitors.edb_telemetry_monitor_utils import utils as etm_utils

# Determine if tests are being run on Github Actions
ON_GITHUB_ACTIONS = '/home/runner' in os.path.expanduser('~') or '/Users/runner' in os.path.expanduser('~')



def test_add_every_change_history():
    """Test that every_change data is correcly combined with an existing
    set of every_change data
    """
    dates1 = np.array([datetime.datetime(2022, 3, 4, 1, 5, i) for i in range(10)])
    data1 = np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2])
    means1 = 0.15
    devs1 = 0.07
    dates2 = np.array([dates1[-1] + datetime.timedelta(seconds=1*i) for i in range(1, 11)])
    data2 = np.array([0.3, 0.4, 0.3, 0.4, 0.3, 0.4, 0.3, 0.4, 0.3, 0.4])
    means2 = 0.35
    devs2 = 0.07
    ec1 = {'0.15': (dates1, data1, means1, devs1),
           '0.35': (dates2, data2, means2, devs2)
           }
    ec2 = {'0.15': (dates1, data1, means1, devs1)}
    combine1 = etm.add_every_change_history(ec1, ec2)
    expected1 = defaultdict(list)
    expected1['0.15'] = (np.append(dates1, dates1), np.append(data1, data1), np.append(means1, means1), np.append(devs1, devs1))
    expected1['0.35'] = (dates2, data2, means2, devs2)

    for key in combine1:
        print('compare ', key)
        for i, cele in enumerate(combine1[key]):
            assert np.all(cele == expected1[key][i])

    dates3 = np.array([dates2[-1] + datetime.timedelta(seconds=1*i) for i in range(1, 11)])
    ec3 = {'0.55': (dates3, data2+0.2, means2+0.2, devs2)}
    combine2 = etm.add_every_change_history(ec1, ec3)
    expected2 = defaultdict(list)
    expected2['0.15'] = (dates1, data1, means1, devs1)
    expected2['0.35'] = (dates2, data2, means2, devs2)
    expected2['0.55'] = (dates3, data2+0.2, means2+0.2, devs2)

    for key in combine2:
        print('compare ', key)
        for i, cele in enumerate(combine2[key]):
            assert np.all(cele == expected2[key][i])


def test_conditions():
    """Test the extraction of data using the ```equal``` class.
    """
    # Create data for mnemonic of interest
    start_time = Time('2022-02-02')
    end_time = Time('2022-02-03')
    temp_data = Table()
    temp_data["euvalues"] = np.array([35., 35.1, 35.2, 36., 36.1, 36.2, 37.1, 37., 36., 36.])
    temp_data["dates"] = np.array([Time('2022-02-02') + TimeDelta(0.1*i, format='jd') for i in range(10)])
    meta = {}
    info = {}
    temperature = EdbMnemonic("TEMPERATURE", start_time, end_time, temp_data, meta, info)

    # Create conditional data
    current_data = {}
    current_data["euvalues"] = np.array([1., 1., 1., 2.5, 2.5, 2.5, 5.5, 5.5, 2.5, 2.5])
    #current_data["dates"] = np.arange(10)*0.1001 + 59612.
    current_data["dates"] = np.array([Time('2022-02-02') + TimeDelta(0.1001*i, format='jd') for i in range(10)])


    ##############################################################
    # Set up condition
    # Using separate classes for each of =, <, >
    #eq25 = cond.equal(current_data, 2.5)
    #condition_list = [eq25]
    #condition_1 = cond.condition(condition_list)


    # pick one of above or below....


    # Or using a single relation class
    eq25 = cond.relation_test(current_data, '==', 2.5)
    condition_list = [eq25]
    condition_1 = cond.condition(condition_list)
    ##########################################################

    # Extract the good data
    condition_1.extract_data(temperature.data)

    # Expected results
    expected_table = Table()
    #expected_table["dates"] = [59612.4, 59612.5, 59612.9]
    frac_days = [0.4, 0.5, 0.9]
    expected_table["dates"] = [Time('2022-02-02') + TimeDelta(frac, format='jd') for frac in frac_days]
    expected_table["euvalues"] = [36.1, 36.2, 36.0]

    assert np.all(condition_1.extracted_data == expected_table)
    assert condition_1.block_indexes == [0, 2]

    grt0 = cond.relation_test(current_data, '>', 0)
    condition_list.append(grt0)
    condition_2 = cond.condition(condition_list)
    condition_2.extract_data(temperature.data)
    assert np.all(condition_2.extracted_data == expected_table)
    assert condition_2.block_indexes == [0, 2]

    less10 = cond.relation_test(current_data, '<', 10)
    condition_list.append(less10)
    condition_3 = cond.condition(condition_list)
    condition_3.extract_data(temperature.data)
    assert np.all(condition_3.extracted_data == expected_table)
    assert condition_3.block_indexes == [0, 2]


def test_every_change_to_allPoints():
    """Make sure we convert every-change data to AllPoints data correctly
    """
    dates = [datetime.datetime(2021, 7, 14, 5, 24, 39+i) for i in range(10)]
    delta = datetime.timedelta(seconds=0.9999)
    values = np.arange(10)
    data = Table([dates, values], names=('dates', 'euvalues'))

    expected_dates = [datetime.datetime(2021, 7, 14, 5, 24, 39+i) for i in range(10)]
    expected_dates = [dates[0]]
    expected_values = [values[0]]
    for i, val in enumerate(values[0:-1]):
        expected_values.extend([val, values[i+1]])
        expected_dates.extend([dates[i] + delta, dates[i+1]])

    expected = Table([expected_dates, expected_values], names=('dates', 'euvalues'))
    updated = etm.every_change_to_allPoints(data)
    assert np.all(expected["dates"] == updated["dates"])
    assert np.all(expected["euvalues"] == updated["euvalues"])

"""
def test_find_all_changes():
    inst = etm.EdbMnemonicMonitor()

    # Create test data
    start_time = Time('2022-02-02')
    end_time = Time('2022-02-03')
    temp_data = Table()
    temp_data["euvalues"] = [350., 350.1, 350.2, 360., 360.1, 360.2, 370.1, 370., 360., 360.]
    #temp_data["dates"] = np.arange(10)*0.1 + 59612.
    temp_data["dates"] = np.array([Time('2022-02-02') + TimeDelta(0.1 * i, format='jd') for i in range(10)])
    meta = {}
    info = {}
    temperature = EdbMnemonic("TEMPERATURE", start_time, end_time, temp_data, meta, info)

    # Create dictionary of dependency info
    dependency = [{"name": "CURRENT", "relation": "none", "threshold": 0}]

    # Create dependency data
    current_data = Table()
    current_data["euvalues"] = [1., 1., 1., 25., 25., 25., 55., 55., 25., 25.]
    #current_data["dates"] = np.arange(10)*0.1001 + 59612.
    current_data["dates"] = np.array([Time('2022-02-02') + TimeDelta(0.1001 * i, format='jd') for i in range(10)])
    inst.query_results[dependency[0]["name"]] = EdbMnemonic("CURRENT", start_time, end_time, current_data, meta, info)


    vals = inst.find_all_changes(temperature, dependency)
    means, meds, devs, times, dep_means, dep_meds, dep_devs, dep_times = vals
    assert means == [35.1, 36.1, 37.05, 36.]
    assert meds == [35.1, 36.1, 37.05, 36.]
    assert devs == []
    assert dep_means == [1., 2., 3., 2.]
    assert dep_meds == [1., 2., 3., 2.]
    assert dep_devs == [0., 0., 0., 0.]
    This is not working well. Strategy for how to find different values in the condition may need to change.
    """

def test_multiple_conditions():
    """Test that filtering using multiple conditions is working as expected.
    """
    # Create data for mnemonic of interest
    start_time = Time('2022-02-02')
    end_time = Time('2022-02-03')
    temp_data = Table()
    temp_data["euvalues"] = Column(np.array([35., 35.1, 35.2, 36., 36.1, 36.2, 37.1, 37., 36., 36.]))
    temp_data["dates"] = Column(np.array([Time('2022-02-02') + TimeDelta(0.1*i, format='jd') for i in range(10)]))
    meta = {}
    info = {}
    temperature = EdbMnemonic("TEMPERATURE", start_time, end_time, temp_data, meta, info)

    # Create conditional data
    current_data = {}
    current_data["euvalues"] = Column(np.array([1., 2.5, 2.5, 2.5, 2.5, 2.5, 5.5, 5.5, 2.5, 2.5]))
    current_data["dates"] = Column(np.array([Time('2022-02-02') + TimeDelta(0.1001*i, format='jd') for i in range(10)]))

    element_data = {}
    element_data["euvalues"] = Column(np.repeat("OFF", 20))
    element_data["euvalues"][13:] = "ON"
    element_data["dates"] = Column(np.array([Time('2022-02-02') + TimeDelta(0.06*i, format='jd') for i in range(20)]))

    grt35 = cond.relation_test(temp_data, '>', 35.11)
    eq25 = cond.relation_test(current_data, '==', 2.5)
    off = cond.relation_test(element_data, '=', 'OFF')
    condition_list = [grt35, eq25, off]
    condition = cond.condition(condition_list)
    condition.extract_data(temperature.data)

    # Compare to expectations
    expected_table = temp_data[2:6]
    assert np.all(condition.extracted_data == expected_table)
    assert condition.block_indexes == [0, 4]


def test_organize_every_change():
    """Test the reorganization of every_change data from an EdbMnemonic into something
    easier to plot
    """
    basetime = Time('2021-04-06 14:00:00')
    dates = np.array([basetime + TimeDelta(600*i, format='sec') for i in range(20)])
    vals = np.array([300.5, 310.3, -250.5, -500.9, 32.2,
                     300.1, 310.8, -250.2, -500.2, 32.7,
                     300.2, 310.4, -250.6, -500.8, 32.3,
                     300.4, 310.5, -250.4, -500.1, 32.9])
    ec_vals = ["F2550W", 'F560W', 'F770W', 'F1000W','F1500W',
               "F2550W", 'F560W', 'F770W', 'F1000W','F1500W',
               "F2550W", 'F560W', 'F770W', 'F1000W','F1500W',
               "F2550W", 'F560W', 'F770W', 'F1000W','F1500W']

    m = Table()
    m["dates"] = dates
    m["euvalues"] = vals
    mnem = EdbMnemonic('IMIR_HK_FW_POS_RATIO', Time('2021-04-06T00:00:00'), Time('2021-04-06T23:00:00'),
                       m, {}, {"unit":"Pos Ratio"})
    mnem.every_change_values = ec_vals
    data = etm.organize_every_change(mnem)

    f2550_idx = [0, 5, 10, 15]
    f560_idx = [1, 6, 11, 16]
    f770_idx = [2, 7, 12, 17]
    f1000_idx = [3, 8, 13, 18]
    f1500_idx = [4, 9, 14, 19]

    f2550_vals = vals[f2550_idx]
    f560_vals = vals[f560_idx]
    f770_vals = vals[f770_idx]
    f1000_vals = vals[f1000_idx]
    f1500_vals = vals[f1500_idx]

    f2550mean, _,  _ = sigma_clipped_stats(f2550_vals, sigma=3)
    f560mean, _,  _ = sigma_clipped_stats(f560_vals, sigma=3)
    f770mean, _,  _ = sigma_clipped_stats(f770_vals, sigma=3)
    f1000mean, _,  _ = sigma_clipped_stats(f1000_vals, sigma=3)
    f1500mean, _,  _ = sigma_clipped_stats(f1500_vals, sigma=3)
    expected = {'F2550W': (np.array([e.datetime for e in dates[f2550_idx]]), f2550_vals, f2550mean),
                'F560W': (np.array([e.datetime for e in dates[f560_idx]]), f560_vals, f560mean),
                'F770W': (np.array([e.datetime for e in dates[f770_idx]]), f770_vals, f770mean),
                'F1000W': (np.array([e.datetime for e in dates[f1000_idx]]), f1000_vals, f1000mean),
                'F1500W': (np.array([e.datetime for e in dates[f1500_idx]]), f1500_vals, f1500mean)}
    for key, val in expected.items():
        assert np.all(val[0] == data[key][0])
        assert np.all(val[1] == data[key][1])
        assert np.all(val[2] == data[key][2])


def test_remove_outer_points():
    """
    """
    data = Table()
    #data["dates"] = [56999.5, 57000., 57000.5, 57001., 57001.5]
    data["dates"] = [Time('2014-12-08') + TimeDelta(0.5 * (i+1), format='jd') for i in range(5)]
    data["euvalues"] = [1, 2, 3, 4, 5]
    mnem = EdbMnemonic('TEST', Time('2022-12-09'), Time('2022-12-10'), data, {}, {})
    etm_utils.remove_outer_points(mnem)
    assert all(mnem.data['MJD'][1:-1] == new.data['dates'])
    assert all(mnem.data['data'][1:-1] == new.data['euvalues'])


def test_get_averaging_time_duration():
    """
    """
    in_strings = ["5_minute", "45_second", "10_day", "2_hour"]
    expected_vals = [5 * u.minute, 45 * u.second, 10 * u.day, 2 * u.hour]

    for inval, outval in zip(in_strings, expected_vals):
        output = etm_utils.get_averaging_time_duration(inval)
        assert output == outval

    bad_strings = ["7_years", "nonsense"]
    for inval in bad_strings:
        with pytest.raises(ValueError) as e_info:
            output = etm_utils.get_averaging_time_duration(inval)


def test_get_query_duration():
    """
    """
    in_strings = ['daily_means', "every_change", "block_means", "time_interval", "none"]
    expected_vals = [15 * u.minute, 1 * u.day, 1 * u.day, 1 * u.day, 1 * u.day,]
    for inval, outval in zip(in_strings, expected_vals):
        output = etm_utils.get_query_duration(inval)
        assert output == outval

    with pytest.raises(ValueError) as e_info:
        output = etm_utils.get_query_duration("bad_string")


def test_key_check():
    """
    """
    d = {'key1': [1,2,3], 'key4': 'a'}
    assert etm_utils.check_key(d, 'key1') == d['key1']
    assert etm_utils.check_key(d, 'key2') == None
