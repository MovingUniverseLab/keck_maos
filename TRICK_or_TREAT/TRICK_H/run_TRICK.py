# run_TRICK.py
# A script to run a MAOS sim of the TRICK sensor potentially with changes
# to the sim parameters like frame rate, guide star magnitude, etc
import numpy as np
import astropy.units as u
import os
from pathlib import Path
import paarti.utils.maos_utils as mu
import reflectao.telemetry_utils as tu
import reflectao.kapa_utils as ku
import reflectao.schema as schema

### Define parameters
out_dir_name = "A_step6_TRICK_H"
seeds = [1]
sim_end_time = 2 * u.s # Jessica reports instability simulating longer than 2s (from 2022, we should check this it might be better now)
lgs_wfs_frame_rate = 500 * u.Hz # LGS WFS frame rate in Hz
tt_wfs_frame_rate = 1000 * u.Hz # Tip tilt WFS frame rate in Hz
lbwfs_int_time = 15 * u.s # We set this equal to 15s because we simulate < 15s and so this doesn't matter. SHould be at least 10s (KAON 1303), with longer integrations for fainter guide stars
lbwfs_focus_frame_rate = tt_wfs_frame_rate / 1 # Because we start observing once the focus has converged, we need to trick the simulation into correcting for focus faster than once every 15s, so we setup a second LBWFS that only corrects focus & tip tilt
# Need to set evl.psf & evl.psfsize to be the same size
zenith_angle = 30 * u.deg # Zenith angle of the observation
# Set the configuration
conf_name = "A_trick_keck_scao_lgs.conf"

### Setup simulation parameters
# Figure out the shortest time interval we need to simulate
lgs_wfs_int_time = (1.0 / lgs_wfs_frame_rate).to(u.s)
tt_int_time = (1.0 / tt_wfs_frame_rate).to(u.s) # STRAP or TRICK are tip tilt (tt) on Keck I. Assuming if TRICK is being used that no light goes to STRAP
lbwfs_focus_frame_time = (1.0 / lbwfs_focus_frame_rate).to(u.s)
max_frame_rate = max(lgs_wfs_frame_rate, tt_wfs_frame_rate) # We don't include the LBWFS or LBWFS focus frame rate here because we know it's slower
sim_dt = (1.0 / max_frame_rate).to(u.s) # Set sim_dt to fastest WFS readout rate in Hz
sim_dtref = sim_dt # sim.dtref defines the reference time over which we provide the signal & background levels. We set it equal to sim.dt for simplicity
sim_end_steps = int((sim_end_time / sim_dt).to_value(u.dimensionless_unscaled))

# Set dtrats (ratio of sample period over dt, must be an integer)
howfs_dtrat = (lgs_wfs_int_time / sim_dt).to_value(u.dimensionless_unscaled)
tt_dtrat = (tt_int_time / sim_dt).to_value(u.dimensionless_unscaled)
lbwfs_dtrat = (lbwfs_int_time / sim_dt).to_value(u.dimensionless_unscaled)
lbwfs_focus_dtrat = (lbwfs_focus_frame_time / sim_dt).to_value(u.dimensionless_unscaled)
powfs_dtrat_array = [int(howfs_dtrat), int(tt_dtrat), int(lbwfs_dtrat), int(lbwfs_focus_dtrat)] # To put in MAOS params

# Check to make sure the dtrats are valid
def check_dtrats(dtrat, dt, name):
    assert dtrat.is_integer(), f"{name} must be an integer. Currently {dtrat}. sim-dt: {dt} and {name}_int_time: {lgs_wfs_int_time if 'howfs' in name else tt_int_time if 'tt' in name else lbwfs_int_time} must be set such that their ratio is an integer."
    assert dtrat > 0, f"{name} must be positive. Currently {dtrat}"
check_dtrats(howfs_dtrat, sim_dt, "howfs_dtrat")
check_dtrats(tt_dtrat, sim_dt, "tt_dtrat")
check_dtrats(lbwfs_dtrat, sim_dt, "lbwfs_dtrat")
check_dtrats(lbwfs_focus_dtrat, sim_dt, "lbwfs_focus_dtrat")


### Final maos command

# cd to the directory with the configurations
maos_configuration_path = Path("/Users/nstieg/work/ao/keck/maos/keck_maos/TRICK_or_TREAT/TRICK/")
os.chdir(maos_configuration_path)

# Define where the output goes

out_dir = Path(f"/Users/nstieg/work/ao/keck/maos/keck_maos/TRICK_or_TREAT/TRICK/outputs/{out_dir_name}/")

# Setup a helper function 
def print_array_maos_style(list_or_array):
    return "[" + " ".join([str(x) for x in list_or_array]) + "]"
pp = print_array_maos_style # So we have array sprinting as [1 2 3] instead of [1, 2, 3] which is what MAOS expects


# Final maos command
maos_cmd = str(f"maos -c {conf_name} -o {out_dir}"
            f" sim.seeds={seeds} sim.zadeg={zenith_angle.to_value(u.deg)}" # include spaces as they're not added automatically
            f" sim.dt={sim_dt.to_value(u.s)} sim.end={sim_end_steps}"
            f" powfs.dtrat={pp(powfs_dtrat_array)}"
            f" save.extra=0" # Prevent segfaults from LBWFS_focus having a dtrat not equal to 1. Fixed in latest dev branch as of 6/8/26
            f" -O")
os.system(maos_cmd)