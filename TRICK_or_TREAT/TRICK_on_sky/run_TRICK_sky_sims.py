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


# Index 0, 1, or 2 for left, middle, or right column of table 1 in the TRICK commissioning KAON 1069 respectively
def run_on_sky_trick_sim(index):

    ### Define parameters
    zenith_angle = 30 * u.deg # ??? 
    r0s = [0.27204656, 0.47932012, 0.12582153] # # seeing of [0.37, 0.21, 0.8] gives r0 of [0.27204656 0.47932012 0.12582153] m from Brooke Digia's fried function in maos_utils as of 6/9/26: r0z1 = 0.976 * ( wvl.to(u.m) / arcsec_to_rad(DIMM).to('', equivalencies=u.dimensionless_angles() ))
    r0zs = [r0 / (np.cos((zenith_angle).to_value(u.radian))**(3/5)) for r0 in r0s] # Convert r0z to r0 at 30 deg zenith angle
    ngs_gs_K_mags = [11.1, 11.5, 13.8] # Left->Right 11.1, 11.5, 13.8
    evl_off_axis_distances = [8.6, 6.39, 0] # Left->Right 8.6, 6.39, 0 arcsec

    # Params to set
    col_names = ["left", "middle", "right"]
    out_dir_name = f"A_table1_{col_names[index]}_col_run3"
    r0z = r0zs[index] * u.m
    tt_wfs_name = "TRICK-K"
    ngs_gs_K_mag = ngs_gs_K_mags[index]
    evl_off_axis_distance = evl_off_axis_distances[index] * u.arcsec

    # Things that should stay the same
    # seeds = [1]
    seeds = [1, 2, 3, 4, 5]
    sim_end_time = 2 * u.s # Jessica reports instability simulating longer than 2s (from 2022, we should check this it might be better now)
    lgs_wfs_frame_rate = 500 * u.Hz # LGS WFS frame rate in Hz
    tt_wfs_frame_rate = 1000 * u.Hz # Tip tilt WFS frame rate in Hz
    lbwfs_int_time = 15 * u.s # We set this equal to 15s because we simulate < 15s and so this doesn't matter. SHould be at least 10s (KAON 1303), with longer integrations for fainter guide stars
    # Need to set evl.psf & evl.psfsize to be the same size
    # Set the configuration
    conf_name = "A_trick_keck_scao_lgs.conf"

    ### Setup simulation parameters
    # Figure out the shortest time interval we need to simulate
    lgs_wfs_int_time = (1.0 / lgs_wfs_frame_rate).to(u.s)
    tt_int_time = (1.0 / tt_wfs_frame_rate).to(u.s) # STRAP or TRICK are tip tilt (tt) on Keck I. Assuming if TRICK is being used that no light goes to STRAP
    max_frame_rate = max(lgs_wfs_frame_rate, tt_wfs_frame_rate) # We don't include the LBWFS or LBWFS focus frame rate here because we know it's slower
    sim_dt = (1.0 / max_frame_rate).to(u.s) # Set sim_dt to fastest WFS readout rate in Hz
    sim_dtref = (1 / 472) * u.s # sim.dtref defines the reference time over which we provide the signal & background levels. We set it equal to sim.dt for simplicity
    sim_end_steps = int((sim_end_time / sim_dt).to_value(u.dimensionless_unscaled))

    # Set dtrats (ratio of sample period over dt, must be an integer)
    howfs_dtrat = (lgs_wfs_int_time / sim_dt).to_value(u.dimensionless_unscaled)
    tt_dtrat = (tt_int_time / sim_dt).to_value(u.dimensionless_unscaled)
    lbwfs_dtrat = (lbwfs_int_time / sim_dt).to_value(u.dimensionless_unscaled)
    powfs_dtrat_array = [int(howfs_dtrat), int(tt_dtrat), int(lbwfs_dtrat)] # To put in MAOS params

    # Check to make sure the dtrats are valid
    def check_dtrats(dtrat, dt, name):
        # Define our own is_integer function that accounts for floating point precision issues
        def is_almost_integer(num):
            return np.isclose(num, int(round(num)))

        # If it's not an integer because of floating point precision, round to the integer it should be, otherwise raise error
        if not dtrat.is_integer():
            if is_almost_integer(dtrat):
                print(f"Warning: {name} is not exactly an integer but is close. Using rounded value.")
                dtrat = int(round(dtrat))
            else:
                raise ValueError(f"{name} must be an integer. Currently {dtrat}. sim-dt: {dt} and {name}_int_time: {lgs_wfs_int_time if 'howfs' in name else tt_int_time if 'tt' in name else lbwfs_int_time} must be set such that their ratio is an integer.")

        # Final assertions
        # assert dtrat.is_integer(), f"{name} must be an integer. Currently {dtrat}. sim-dt: {dt} and {name}_int_time: {lgs_wfs_int_time if 'howfs' in name else tt_int_time if 'tt' in name else lbwfs_int_time} must be set such that their ratio is an integer."
        assert dtrat > 0, f"{name} must be positive. Currently {dtrat}"


    check_dtrats(howfs_dtrat, sim_dt, "howfs_dtrat")
    check_dtrats(tt_dtrat, sim_dt, "tt_dtrat")
    check_dtrats(lbwfs_dtrat, sim_dt, "lbwfs_dtrat")

    # Set noise equivalent angle of the WFS based on gs mag (@ dtref of 1/472)
    howfs_nearecon = 43 # Doesn't matter because we're using physical optics not geometric optics but need something to pass in
    r0_assumed = r0s[index]
    SNR, sigma_theta, Np, Nb = mu.keck_nea_photons(ngs_gs_K_mag, tt_wfs_name, r0_assumed, wfs_int_time=sim_dtref.to_value(u.s))
    tt_nearecon = (sigma_theta * u.mas).to_value(u.mas)
    lbwfs_nearecon = 1.28 # Doesn't matter because we're using physical optics not geometric optics but need something to pass in
    nearecon = [howfs_nearecon, tt_nearecon, lbwfs_nearecon]

    ### Final maos command

    # cd to the directory with the configurations
    maos_configuration_path = Path("/Users/nstieg/work/ao/keck/maos/keck_maos/TRICK_or_TREAT/TRICK_on_sky/")
    os.chdir(maos_configuration_path)

    # Define where the output goes

    out_dir = Path(f"/Users/nstieg/work/ao/keck/maos/keck_maos/TRICK_or_TREAT/TRICK_on_sky/outputs/{out_dir_name}/")

    # Setup a helper function 
    def print_array_maos_style(list_or_array):
        return "[" + " ".join([str(x) for x in list_or_array]) + "]"
    pp = print_array_maos_style # So we have array sprinting as [1 2 3] instead of [1, 2, 3] which is what MAOS expects


    # Final maos command
    maos_cmd = str(f"maos -c {conf_name} -o {out_dir}"
                f" sim.seeds={seeds} sim.zadeg={zenith_angle.to_value(u.deg)}" # include spaces as they're not added automatically
                f" sim.dt={sim_dt.to_value(u.s)} sim.end={sim_end_steps}"
                f" powfs.dtrat={pp(powfs_dtrat_array)} powfs.nearecon={pp(nearecon)}"
                f" atm.r0z={r0z.to_value(u.m)}"
                f" evl.thetax=[{evl_off_axis_distance.to_value(u.arcsec)}]"
                f" save.extra=0" # Prevent segfaults from LBWFS_focus having a dtrat not equal to 1. Fixed in latest dev branch as of 6/8/26
                f" -O")
    os.system(maos_cmd)

if __name__ == "__main__":
    for index in range(3):
        run_on_sky_trick_sim(index)