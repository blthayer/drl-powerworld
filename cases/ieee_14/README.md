# ieee_14
IEEE 14 bus model. The following files were downloaded from [Texas A&M](https://electricgrids.engr.tamu.edu/electric-grid-test-cases/ieee-14-bus-system/)
on 2020-01-27:
- IEEE 14 bus.epc
- IEEE 14 bus.pwb
- IEEE 14 bus.pwd
- IEEE 14 bus.raw

The following files were pulled from the [gym-powerworld](https://github.com/blthayer/gym-powerworld)
repository at commit [5540b2704450acb87fee7f57f6766a8bea419f06](https://github.com/blthayer/gym-powerworld/tree/5540b2704450acb87fee7f57f6766a8bea419f06):
- contour.axd
- IEEE 14 bus.axd
- IEEE 14 bus condensers.PWB

The .axd files were generated via the PowerWorld UI, and are needed for
visualizing the environment. The "condensers" file has the high and low 
MW limits for generators at buses 3, 6, and 8 set to 0 in order to
both replicate what the original case is supposed to be as well as 
replicate how the GridMind team handled the case.