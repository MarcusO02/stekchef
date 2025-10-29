using System;
using System.IO;
using System.Linq;
using System.Drawing;
using System.Drawing.Imaging;
using Microsoft.Kinect;

class Recorder
{
    static KinectSensor sensor;

    static void Main(string[] args)
    {
        // 1) Output directory
        string outDir = args.Length > 0 ? args[0] : "frames_out";
        Directory.CreateDirectory(outDir);

        // 2) Find a connected Kinect v1
        sensor = KinectSensor.KinectSensors
                    .Cast<KinectSensor>()
                    .FirstOrDefault(s => s != null && s.Status == KinectStatus.Connected);

        if (sensor == null)
        {
            Console.WriteLine("No Kinect v1 detected by Microsoft SDK 1.8.\n" +
                              "Check drivers/power/USB. Then rerun.");
            Console.ReadLine();
            return;
        }

        // 3) Enable streams (color is 32 bpp; depth is 16-bit + player index internally)
        sensor.ColorStream.Enable(ColorImageFormat.RgbResolution640x480Fps30);
        sensor.DepthStream.Enable(DepthImageFormat.Resolution640x480Fps30);

        // Try to enable Near Range (only works on Kinect for Windows v1, not Xbox models)
        try
        {
            sensor.DepthStream.Range = DepthRange.Near;
            Console.WriteLine("Near Range enabled (minimum distance ~400 mm).");
        }
        catch (InvalidOperationException)
        {
            Console.WriteLine("Near Range not supported on this device (likely Xbox Kinect, min ~800 mm).");
        }

        var mapper = sensor.CoordinateMapper;

        // 4) Frame callback
        sensor.AllFramesReady += (s, e) =>
        {
            using (var c = e.OpenColorImageFrame())
            using (var d = e.OpenDepthImageFrame())
            {
                if (c == null || d == null) return;

                long t = DateTime.UtcNow.Ticks;

                // ---- COLOR: save as PNG (32 bpp) ----
                try
                {
                    byte[] color = new byte[c.PixelDataLength]; // 640*480*4 (BGRX)
                    c.CopyPixelDataTo(color);

                    using (var bmp = new Bitmap(c.Width, c.Height, PixelFormat.Format32bppRgb))
                    {
                        var rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
                        var bd = bmp.LockBits(rect, ImageLockMode.WriteOnly, bmp.PixelFormat);
                        System.Runtime.InteropServices.Marshal.Copy(color, 0, bd.Scan0, color.Length);
                        bmp.UnlockBits(bd);

                        string rgbPath = Path.Combine(outDir, $"rgb_{t}.png");
                        bmp.Save(rgbPath, ImageFormat.Png);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Failed to save RGB frame: " + ex.Message);
                }

                // ---- DEPTH (ORIGINAL): save raw uint16 mm + sidecar (using DepthImagePixel[]) ----
                try
                {
                    int depthW = d.Width, depthH = d.Height;
                    DepthImagePixel[] depthPixels = new DepthImagePixel[d.PixelDataLength];
                    d.CopyDepthImagePixelDataTo(depthPixels);

                    // Convert to a contiguous ushort[] of millimeters
                    ushort[] depthMm = new ushort[depthPixels.Length];
                    for (int i = 0; i < depthPixels.Length; i++)
                        depthMm[i] = (ushort)depthPixels[i].Depth; // 0 = invalid

                    string basePath = Path.Combine(outDir, $"depth_{t}");
                    byte[] bytes = new byte[depthMm.Length * sizeof(ushort)];
                    Buffer.BlockCopy(depthMm, 0, bytes, 0, bytes.Length);
                    File.WriteAllBytes(basePath + ".depth16le", bytes);

                    File.WriteAllText(basePath + ".shape.txt",
                        $"{depthW} {depthH} 1 mm uint16le");
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Failed to save original depth frame: " + ex.Message);
                }

                // ---- DEPTH ALIGNED TO RGB: save uint16 mm mapped into color image ----
                try
                {
                    // Copy depth as DepthImagePixel[] (again or reuse above if you keep it)
                    DepthImagePixel[] depthPixels = new DepthImagePixel[d.PixelDataLength];
                    d.CopyDepthImagePixelDataTo(depthPixels);

                    // Map to color space
                    var colorPoints = new ColorImagePoint[depthPixels.Length];
                    mapper.MapDepthFrameToColorFrame(
                        d.Format,
                        depthPixels,
                        sensor.ColorStream.Format,
                        colorPoints
                    );

                    int colorW = c.Width, colorH = c.Height;
                    ushort[] depthAligned = new ushort[colorW * colorH]; // 0 = invalid

                    int depthW = d.Width, depthH = d.Height;
                    for (int dy = 0; dy < depthH; dy++)
                    {
                        int rowOffset = dy * depthW;
                        for (int dx = 0; dx < depthW; dx++)
                        {
                            int di = rowOffset + dx;
                            int depthMm = depthPixels[di].Depth;
                            if (depthMm <= 0) continue; // skip invalid

                            var cp = colorPoints[di];
                            int cx = cp.X, cy = cp.Y;

                            if ((uint)cx < (uint)colorW && (uint)cy < (uint)colorH)
                            {
                                int ci = cy * colorW + cx;
                                ushort current = depthAligned[ci];
                                if (current == 0 || depthMm < current)
                                    depthAligned[ci] = (ushort)depthMm; // keep nearest
                            }
                        }
                    }

                    string baseAligned = Path.Combine(outDir, $"depth_aligned_{t}");
                    byte[] bytesAligned = new byte[depthAligned.Length * sizeof(ushort)];
                    Buffer.BlockCopy(depthAligned, 0, bytesAligned, 0, bytesAligned.Length);
                    File.WriteAllBytes(baseAligned + ".depth16le", bytesAligned);
                    File.WriteAllText(baseAligned + ".shape.txt",
                        $"{colorW} {colorH} 1 mm uint16le");

                    Console.WriteLine($"Saved RGB + depth + depth_aligned at {t}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Failed to compute/save aligned depth: " + ex.Message);
                }
            }
        };

        // 5) Start/stop sensor
        try
        {
            sensor.Start();
            Console.WriteLine("Recording... press Enter to stop.");
            Console.ReadLine();
        }
        catch (System.IO.IOException)
        {
            Console.WriteLine("Kinect is already in use by another process. Close other apps and try again.");
            Console.ReadLine();
        }
        catch (Exception ex)
        {
            Console.WriteLine("Failed to start Kinect sensor: " + ex.Message);
            Console.ReadLine();
        }
        {
            if (sensor != null && sensor.IsRunning) sensor.Stop();
        }
    }
}
