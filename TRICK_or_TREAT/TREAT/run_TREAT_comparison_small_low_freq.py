# run_TREAT.py
# A script to run a MAOS sim of the TREAT sensor potentially with changes
# to the sim parameters like frame rate, guide star magnitude, etc
import numpy as np
import astropy.units as u
import os
import sys
import subprocess
from pathlib import Path
import paarti.utils.maos_utils as mu
import reflectao.telemetry_utils as tu
import reflectao.kapa_utils as ku
import reflectao.schema as schema


def run_one_TREAT_sim(out_dir_name, ngs_gs_mag, tt_wfs_frame_rate=3500 * u.Hz):

    ### Define parameters
    # out_dir_name = "A_step12_3500Hz_low_rn_big_pixels"
    # tt_wfs_frame_rate = 1000 * u.Hz # Tip tilt WFS frame rate in Hz
    conf_name = "A_treat_keck_scao_lgs.conf"
    tt_wfs_name = "TREAT-K" # or TRICK-K or STRAP
    # ngs_gs_mag = 8.3 # 8.3 is the NGS magnitude used for keck_base

    # Should stay the same
    # seeds = [1, 2, 3, 4, 5]
    seeds = [1]
    sim_end_time = 2 * u.s # Jessica reports instability simulating longer than 2s (from 2022, we should check this it might be better now)
    lgs_wfs_frame_rate = 500 * u.Hz # LGS WFS frame rate in Hz
    lbwfs_int_time = 15 * u.s # We set this equal to 15s because we simulate < 15s and so this doesn't matter. SHould be at least 10s (KAON 1303), with longer integrations for fainter guide stars
    # Need to set evl.psf & evl.psfsize to be the same size
    zenith_angle = 30 * u.deg # Zenith angle of the observation
    # Set the configuration

    ### Setup simulation parameters
    # Figure out the shortest time interval we need to simulate
    lgs_wfs_int_time = (1.0 / lgs_wfs_frame_rate).to(u.s)
    tt_int_time = (1.0 / tt_wfs_frame_rate).to(u.s)
    max_frame_rate = max(lgs_wfs_frame_rate, tt_wfs_frame_rate) # We don't include the LBWFS or LBWFS focus frame rate here because we know it's slower
    sim_dt = (1.0 / max_frame_rate).to(u.s) # Set sim_dt to fastest WFS readout rate in Hz
    sim_dtref = (1 / (472 * u.Hz)).to(u.s) # sim.dtref defines the reference time over which we provide the signal & background levels. We've been using 1/472 to match keck_base
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
    r0z_assumed = 0.14 # Doesn't affect our NEA calculations here, but we need to provide something for the function call
    r0_assumed = r0z_assumed * np.cos(zenith_angle.to_value(u.radian))**(3/5)
    SNR, sigma_theta, Np, Nb = mu.keck_nea_photons(ngs_gs_mag, tt_wfs_name, r0_assumed, wfs_int_time=sim_dtref.to_value(u.s))
    tt_nearecon = (sigma_theta * u.mas).to_value(u.mas)
    lbwfs_nearecon = 1.28 # Doesn't matter because we're using physical optics not geometric optics but need something to pass in
    nearecon = [howfs_nearecon, tt_nearecon, lbwfs_nearecon]

    ### Final maos command

    # cd to the directory with the configurations
    maos_configuration_path = Path("/Users/nstieg/work/ao/keck/maos/keck_maos/TRICK_or_TREAT/TREAT/")
    os.chdir(maos_configuration_path)

    # Define where the output goes

    out_dir = Path(f"/Users/nstieg/work/ao/keck/maos/keck_maos/TRICK_or_TREAT/TREAT/outputs/{out_dir_name}/")

    # Setup a helper function 
    def print_array_maos_style(list_or_array):
        return "[" + " ".join([str(x) for x in list_or_array]) + "]"
    pp = print_array_maos_style # So we have array sprinting as [1 2 3] instead of [1, 2, 3] which is what MAOS expects


    # Final maos command
    maos_cmd = str(f"maos -c {conf_name} -o {out_dir}"
                f" sim.seeds={seeds} sim.zadeg={zenith_angle.to_value(u.deg)}" # include spaces as they're not added automatically
                f" sim.dt={sim_dt.to_value(u.s)} sim.end={sim_end_steps}"
                f" powfs.dtrat={pp(powfs_dtrat_array)} powfs.nearecon={pp(nearecon)}"
                f" save.extra=0" # Prevent segfaults from LBWFS_focus having a dtrat not equal to 1. Fixed in latest dev branch as of 6/8/26
                f" -O")
    
    # os.system(maos_cmd)
    try:
        subprocess.run(maos_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nMAOS execution failed on magnitude {ngs_gs_mag} with exit code {e.returncode}!", file=sys.stderr)
        print("Stopping loop execution to prevent faulty data collection.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    for ngs_gs_mag in np.arange(8, 18.5, 1):
    # for ngs_gs_mag in [8.0]:
        tt_wfs_frame_rate = 1000 * u.Hz # Tip tilt WFS frame rate in Hz
        out_dir_name = f"A_comparison_tt_mag_{ngs_gs_mag}_run5_small_{tt_wfs_frame_rate.to_value(u.Hz):.0f}Hz"
        run_one_TREAT_sim(out_dir_name, ngs_gs_mag, tt_wfs_frame_rate)