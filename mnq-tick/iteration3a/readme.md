Experimenting with iter3.

changes in export:
- removed adpVel (needs cfb), bollinger bands
- rsx(15,open)->rsx(16,hlc)
- vel(16,jma)->vel(16,hlc)


- add 5th stream
    - RSX
    - VEL
- back to 4 streams - replace TICK_D2 with the above
- 2 sec bars