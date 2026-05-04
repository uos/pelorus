#!/usr/bin/env python3
import os, struct
TAG_IMU=1
TAG_CLOUD=2
NUM_POINTS=512

def write_imu(f,i):
    t=1000.0+i*0.005
    gx=0.004+0.001*(i%3-1)
    gy=-0.002+0.001*((i+1)%3-1)
    gz=0.010+0.002*((i+2)%3-1)
    ax=0.03+0.01*(i%5-2)
    ay=-0.015+0.01*((i+1)%5-2)
    az=9.81+0.02+0.01*((i+2)%5-2)
    f.write(struct.pack('<Bdffffff',TAG_IMU,t,gx,gy,gz,ax,ay,az))

def base_point(i):
    if i<NUM_POINTS//2:
        u=i%16; v=i//16
        return (4.0+0.12*u,-1.0+0.12*v,0.01*((u+v)%2))
    j=i-NUM_POINTS//2
    u=j%16; v=j//16
    return (6.0+0.005*((u+v)%2),-1.0+0.12*u,0.10+0.12*v)

def write_cloud(f,start,end,moved):
    f.write(struct.pack('<Bddi',TAG_CLOUD,start,end,NUM_POINTS))
    c=0.996802; s=0.079915
    for i in range(NUM_POINTS):
        x,y,z=base_point(i)
        if moved:
            x,y,z=c*x-s*y+0.18,s*x+c*y-0.12,z+0.03
        intensity=0.0
        t_rel_sec=i/NUM_POINTS*0.1
        f.write(struct.pack('<fffff',x,y,z,intensity,t_rel_sec))

def main():
    out='synthetic_dataset_nomotion.bin'
    with open(out,'wb') as f:
        for i in range(200): write_imu(f,i)
        write_cloud(f,1000.6,1000.7,False)
        for i in range(200,300): write_imu(f,i)
        write_cloud(f,1001.2,1001.3,False)
    print(out)

if __name__=='__main__':
    main()
