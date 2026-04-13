/* record_sequence.c — capture a live pelorus_c run to a binary file
 *
 * Reads IMU and cloud data from stdin in a simple text format and
 * simultaneously feeds it to pelorus_c while writing the binary
 * sequence that equality_test.c can replay.
 *
 * Build (from repo root):
 *   cargo build -p pelorus_c --release
 *
 *   gcc -O2 -std=c99 \
 *       -Ipelorus_c \
 *       pelorus_pico/tests/record_sequence.c \
 *       -Ltarget/release -lpelorus_c \
 *       -lpthread -ldl -lm \
 *       -Wl,-rpath,target/release \
 *       -o record_seq
 *
 * Usage:
 *   ./record_seq output.bin < input.txt
 *   # or pipe from a live source
 *
 * Text input format (one record per line):
 *   IMU  <t> <gx> <gy> <gz> <ax> <ay> <az>
 *   CLOUD <t_start> <t_end> <n>
 *   <x0> <y0> <z0> <i0> <ts0>
 *   ...   (n lines of 5 floats: x y z intensity timestamp_rel_seconds)
 *
 * Binary output format is identical to what equality_test.c reads:
 *   TAG_IMU  (1) + double t + 6×float
 *   TAG_CLOUD(2) + double t_start + double t_end + int32_t n
 *               + n × 5×float
 *
 * This utility is optional — you can also generate the binary file from
 * any other source (e.g., a ROS bag reader or a custom sensor driver).
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define TAG_IMU   ((uint8_t)1)
#define TAG_CLOUD ((uint8_t)2)

static void write_u8(FILE *f, uint8_t v)  { fwrite(&v, 1, 1, f); }
static void write_f64(FILE *f, double v)  { fwrite(&v, sizeof(double), 1, f); }
static void write_f32(FILE *f, float  v)  { fwrite(&v, sizeof(float),  1, f); }
static void write_i32(FILE *f, int32_t v) { fwrite(&v, sizeof(int32_t), 1, f); }

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <output.bin>\n", argv[0]);
        return 1;
    }

    FILE *out = fopen(argv[1], "wb");
    if (!out) { perror("fopen"); return 1; }

    char line[4096];
    int  n_imu = 0, n_cloud = 0;

    while (fgets(line, sizeof(line), stdin)) {
        char tag[16];
        if (sscanf(line, "%15s", tag) != 1) continue;

        if (strcmp(tag, "IMU") == 0) {
            double t;
            float  gx, gy, gz, ax, ay, az;
            if (sscanf(line, "%*s %lf %f %f %f %f %f %f",
                       &t, &gx, &gy, &gz, &ax, &ay, &az) != 7) {
                fprintf(stderr, "Malformed IMU line: %s", line);
                continue;
            }
            write_u8(out, TAG_IMU);
            write_f64(out, t);
            write_f32(out, gx); write_f32(out, gy); write_f32(out, gz);
            write_f32(out, ax); write_f32(out, ay); write_f32(out, az);
            n_imu++;

        } else if (strcmp(tag, "CLOUD") == 0) {
            double  t_start, t_end;
            int32_t n;
            if (sscanf(line, "%*s %lf %lf %d", &t_start, &t_end, &n) != 3) {
                fprintf(stderr, "Malformed CLOUD line: %s", line);
                continue;
            }

            /* Read n point lines first, buffer them. */
            float *buf = (float *)malloc((size_t)n * 5 * sizeof(float));
            if (!buf) { fprintf(stderr, "malloc failed\n"); fclose(out); return 1; }

            int i, ok = 1;
            for (i = 0; i < n; i++) {
                if (!fgets(line, sizeof(line), stdin)) { ok = 0; break; }
                if (sscanf(line, "%f %f %f %f %f",
                           &buf[i*5+0], &buf[i*5+1], &buf[i*5+2],
                           &buf[i*5+3], &buf[i*5+4]) != 5) {
                    fprintf(stderr, "Malformed point line %d: %s", i, line);
                    ok = 0; break;
                }
            }
            if (!ok) { free(buf); break; }

            write_u8(out,  TAG_CLOUD);
            write_f64(out, t_start);
            write_f64(out, t_end);
            write_i32(out, n);
            for (i = 0; i < n; i++) {
                write_f32(out, buf[i*5+0]);
                write_f32(out, buf[i*5+1]);
                write_f32(out, buf[i*5+2]);
                write_f32(out, buf[i*5+3]);
                write_f32(out, buf[i*5+4]);
            }
            free(buf);
            n_cloud++;
        }
        /* skip any unrecognised lines */
    }

    fclose(out);
    printf("Wrote %d IMU + %d cloud records to %s\n", n_imu, n_cloud, argv[1]);
    return 0;
}
