"""
ComfyUI Pixel Art Scaler Node
Pure Python implementation of pixel art scaling algorithms
No external dependencies required beyond ComfyUI's standard libraries

Place this file in: ComfyUI/custom_nodes/pixel_art_scaler/__init__.py
"""

import numpy as np
import torch
import folder_paths
import comfy.utils

class PixelArtScaler:
    """
    A ComfyUI node for scaling pixel art using various algorithms
    Pure Python implementation - no external dependencies
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "algorithm": ([
                    "2xSaI",
                    "Eagle2x", "Eagle3x", "Eagle4x",
                    "HQ2x", "HQ3x", "HQ4x",
                    "NearestNeighbor2x", "NearestNeighbor3x", "NearestNeighbor4x",
                    "Scale2x", "Scale3x",
                    "Super2xSaI",
                    "SuperEagle",
                    "xBR2x", "xBR3x", "xBR4x"
                ],),
                "threshold": ("FLOAT", {
                    "default": 48.0,
                    "min": 0.0,
                    "max": 255.0,
                    "step": 1.0,
                    "display": "slider"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "scale_pixel_art"
    CATEGORY = "image/upscaling"
    
    def rgb_to_yuv(self, r, g, b):
        """Convert RGB to YUV for better color comparison"""
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v = 0.615 * r - 0.51499 * g - 0.10001 * b
        return y, u, v
    
    def color_diff(self, c1, c2):
        """Calculate color difference using YUV space"""
        if c1 is None or c2 is None:
            return 999999
        
        # Handle numpy arrays properly
        r1 = float(c1[0]) * 255
        g1 = float(c1[1]) * 255
        b1 = float(c1[2]) * 255
        r2 = float(c2[0]) * 255
        g2 = float(c2[1]) * 255
        b2 = float(c2[2]) * 255
        
        y1, u1, v1 = self.rgb_to_yuv(r1, g1, b1)
        y2, u2, v2 = self.rgb_to_yuv(r2, g2, b2)
        
        return abs(y1 - y2) + abs(u1 - u2) + abs(v1 - v2)
    
    def pixels_equal(self, c1, c2, threshold):
        """Check if two pixels are similar within threshold"""
        if c1 is None or c2 is None:
            return False
        return self.color_diff(c1, c2) < threshold
    
    def get_pixel_safe(self, img, y, x):
        """Safely get pixel value with boundary checking"""
        h, w = img.shape[:2]
        if y < 0 or y >= h or x < 0 or x >= w:
            return None
        return img[y, x].copy()  # Return a copy to avoid reference issues
    
    def scale2x_core(self, img, threshold):
        """Scale2x algorithm implementation"""
        h, w = img.shape[:2]
        out = np.zeros((h * 2, w * 2, img.shape[2]), dtype=np.float32)
        
        for y in range(h):
            for x in range(w):
                # Get center pixel
                e = img[y, x].copy()
                
                # Get surrounding pixels
                a = self.get_pixel_safe(img, y - 1, x)
                b = self.get_pixel_safe(img, y, x - 1) 
                c = self.get_pixel_safe(img, y, x + 1)
                d = self.get_pixel_safe(img, y + 1, x)
                
                # If any neighbor is missing, use center pixel
                if a is None: a = e.copy()
                if b is None: b = e.copy()
                if c is None: c = e.copy()
                if d is None: d = e.copy()
                
                # Scale2x rules
                e0 = e.copy()
                e1 = e.copy()
                e2 = e.copy()
                e3 = e.copy()
                
                bd_diff = self.color_diff(b, d)
                ac_diff = self.color_diff(a, c)
                
                if bd_diff < threshold and bd_diff < ac_diff:
                    if self.pixels_equal(a, b, threshold):
                        e0 = b.copy()
                    if self.pixels_equal(b, c, threshold):
                        e1 = c.copy()
                    if self.pixels_equal(d, a, threshold):
                        e2 = b.copy()
                    if self.pixels_equal(c, d, threshold):
                        e3 = c.copy()
                
                # Set output pixels
                out[y * 2, x * 2] = e0
                out[y * 2, x * 2 + 1] = e1
                out[y * 2 + 1, x * 2] = e2
                out[y * 2 + 1, x * 2 + 1] = e3
        
        return out
    
    def hq2x_core(self, img, threshold):
        """HQ2x algorithm implementation"""
        h, w = img.shape[:2]
        out = np.zeros((h * 2, w * 2, img.shape[2]), dtype=np.float32)
        
        for y in range(h):
            for x in range(w):
                # Get 3x3 neighborhood
                c4 = self.get_pixel_safe(img, y - 1, x - 1)
                c1 = self.get_pixel_safe(img, y - 1, x)
                c5 = self.get_pixel_safe(img, y - 1, x + 1)
                c3 = self.get_pixel_safe(img, y, x - 1)
                c0 = img[y, x].copy()  # center
                c2 = self.get_pixel_safe(img, y, x + 1)
                c6 = self.get_pixel_safe(img, y + 1, x - 1)
                c7 = self.get_pixel_safe(img, y + 1, x)
                c8 = self.get_pixel_safe(img, y + 1, x + 1)
                
                # Replace None with center pixel
                if c1 is None: c1 = c0.copy()
                if c2 is None: c2 = c0.copy()
                if c3 is None: c3 = c0.copy()
                if c4 is None: c4 = c0.copy()
                if c5 is None: c5 = c0.copy()
                if c6 is None: c6 = c0.copy()
                if c7 is None: c7 = c0.copy()
                if c8 is None: c8 = c0.copy()
                
                # Determine pattern
                pattern = 0
                if self.pixels_equal(c1, c0, threshold): pattern |= 1
                if self.pixels_equal(c2, c0, threshold): pattern |= 2
                if self.pixels_equal(c7, c0, threshold): pattern |= 4
                if self.pixels_equal(c3, c0, threshold): pattern |= 8
                
                # Output position
                x2 = x * 2
                y2 = y * 2
                
                # Apply HQ2x interpolation based on pattern
                if pattern == 0 or pattern == 15:
                    # No edges or all edges - keep original
                    out[y2, x2] = c0
                    out[y2, x2 + 1] = c0
                    out[y2 + 1, x2] = c0
                    out[y2 + 1, x2 + 1] = c0
                elif pattern == 1:  # Top edge
                    out[y2, x2] = (c0 * 3 + c1) / 4
                    out[y2, x2 + 1] = (c0 * 3 + c1) / 4
                    out[y2 + 1, x2] = c0
                    out[y2 + 1, x2 + 1] = c0
                elif pattern == 2:  # Right edge
                    out[y2, x2] = c0
                    out[y2, x2 + 1] = (c0 * 3 + c2) / 4
                    out[y2 + 1, x2] = c0
                    out[y2 + 1, x2 + 1] = (c0 * 3 + c2) / 4
                elif pattern == 4:  # Bottom edge
                    out[y2, x2] = c0
                    out[y2, x2 + 1] = c0
                    out[y2 + 1, x2] = (c0 * 3 + c7) / 4
                    out[y2 + 1, x2 + 1] = (c0 * 3 + c7) / 4
                elif pattern == 8:  # Left edge
                    out[y2, x2] = (c0 * 3 + c3) / 4
                    out[y2, x2 + 1] = c0
                    out[y2 + 1, x2] = (c0 * 3 + c3) / 4
                    out[y2 + 1, x2 + 1] = c0
                elif pattern == 3:  # Top-right corner
                    out[y2, x2] = (c0 * 2 + c1 + c3) / 4
                    out[y2, x2 + 1] = (c0 * 2 + c1 + c2) / 4
                    out[y2 + 1, x2] = c0
                    out[y2 + 1, x2 + 1] = (c0 * 3 + c2) / 4
                elif pattern == 6:  # Bottom-right corner
                    out[y2, x2] = c0
                    out[y2, x2 + 1] = (c0 * 3 + c2) / 4
                    out[y2 + 1, x2] = (c0 * 2 + c7 + c3) / 4
                    out[y2 + 1, x2 + 1] = (c0 * 2 + c7 + c2) / 4
                elif pattern == 9:  # Top-left corner
                    out[y2, x2] = (c0 * 2 + c1 + c3) / 4
                    out[y2, x2 + 1] = (c0 * 2 + c1 + c2) / 4
                    out[y2 + 1, x2] = (c0 * 3 + c3) / 4
                    out[y2 + 1, x2 + 1] = c0
                elif pattern == 12:  # Bottom-left corner
                    out[y2, x2] = (c0 * 3 + c3) / 4
                    out[y2, x2 + 1] = c0
                    out[y2 + 1, x2] = (c0 * 2 + c7 + c3) / 4
                    out[y2 + 1, x2 + 1] = (c0 * 2 + c7 + c2) / 4
                else:
                    # Complex patterns - blend appropriately
                    out[y2, x2] = (c0 * 2 + c1 + c3) / 4
                    out[y2, x2 + 1] = (c0 * 2 + c1 + c2) / 4
                    out[y2 + 1, x2] = (c0 * 2 + c7 + c3) / 4
                    out[y2 + 1, x2 + 1] = (c0 * 2 + c7 + c2) / 4
        
        return out
    
    def xbr2x_core(self, img, threshold):
        """xBR 2x algorithm implementation"""
        h, w = img.shape[:2]
        out = np.zeros((h * 2, w * 2, img.shape[2]), dtype=np.float32)
        
        for y in range(h):
            for x in range(w):
                # Get 5x5 neighborhood for xBR
                # We need a larger kernel for xBR
                p = {}
                
                # Define the 21 pixels used in xBR
                p['a1'] = self.get_pixel_safe(img, y - 2, x - 1)
                p['b1'] = self.get_pixel_safe(img, y - 1, x - 2)
                p['a'] = self.get_pixel_safe(img, y - 1, x - 1)
                p['b'] = self.get_pixel_safe(img, y - 1, x)
                p['c'] = self.get_pixel_safe(img, y - 1, x + 1)
                p['c4'] = self.get_pixel_safe(img, y - 2, x + 1)
                p['d'] = self.get_pixel_safe(img, y, x - 1)
                p['e'] = img[y, x].copy()  # center
                p['f'] = self.get_pixel_safe(img, y, x + 1)
                p['f4'] = self.get_pixel_safe(img, y - 1, x + 2)
                p['g'] = self.get_pixel_safe(img, y + 1, x - 1)
                p['h'] = self.get_pixel_safe(img, y + 1, x)
                p['i'] = self.get_pixel_safe(img, y + 1, x + 1)
                p['i4'] = self.get_pixel_safe(img, y + 2, x + 1)
                p['g5'] = self.get_pixel_safe(img, y + 1, x - 2)
                p['h5'] = self.get_pixel_safe(img, y + 2, x - 1)
                p['i5'] = self.get_pixel_safe(img, y + 2, x)
                
                # Replace None values with center pixel
                e = p['e']
                for key in p:
                    if p[key] is None:
                        p[key] = e.copy()
                
                # xBR uses complex edge detection
                # Simplified version for basic implementation
                a = p['a']
                b = p['b']
                c = p['c']
                d = p['d']
                f = p['f']
                g = p['g']
                h = p['h']
                i = p['i']
                
                # Output position
                x2 = x * 2
                y2 = y * 2
                
                # Initialize output with center pixel
                e0 = e.copy()
                e1 = e.copy()
                e2 = e.copy()
                e3 = e.copy()
                
                # xBR edge detection and interpolation
                # Check for diagonal edges
                if self.pixels_equal(d, b, threshold) and not self.pixels_equal(a, e, threshold) and not self.pixels_equal(e, c, threshold):
                    e0 = (d + b) / 2
                
                if self.pixels_equal(b, f, threshold) and not self.pixels_equal(c, e, threshold) and not self.pixels_equal(e, a, threshold):
                    e1 = (b + f) / 2
                
                if self.pixels_equal(h, d, threshold) and not self.pixels_equal(g, e, threshold) and not self.pixels_equal(e, a, threshold):
                    e2 = (h + d) / 2
                
                if self.pixels_equal(f, h, threshold) and not self.pixels_equal(i, e, threshold) and not self.pixels_equal(e, c, threshold):
                    e3 = (f + h) / 2
                
                # Additional smoothing for better results
                # Check for L-shapes
                if self.pixels_equal(b, d, threshold):
                    diff_ab = self.color_diff(a, b)
                    diff_be = self.color_diff(b, e)
                    if diff_ab < diff_be:
                        e0 = (a + b + d + e) / 4
                
                if self.pixels_equal(b, f, threshold):
                    diff_bc = self.color_diff(b, c)
                    diff_be = self.color_diff(b, e)
                    if diff_bc < diff_be:
                        e1 = (b + c + e + f) / 4
                
                if self.pixels_equal(d, h, threshold):
                    diff_dg = self.color_diff(d, g)
                    diff_de = self.color_diff(d, e)
                    if diff_dg < diff_de:
                        e2 = (d + e + g + h) / 4
                
                if self.pixels_equal(f, h, threshold):
                    diff_fi = self.color_diff(f, i)
                    diff_fe = self.color_diff(f, e)
                    if diff_fi < diff_fe:
                        e3 = (e + f + h + i) / 4
                
                # Set output pixels
                out[y2, x2] = e0
                out[y2, x2 + 1] = e1
                out[y2 + 1, x2] = e2
                out[y2 + 1, x2 + 1] = e3
        
        return out
    
    def eagle2x_core(self, img, threshold):
        """Eagle 2x algorithm implementation"""
        h, w = img.shape[:2]
        out = np.zeros((h * 2, w * 2, img.shape[2]), dtype=np.float32)
        
        for y in range(h):
            for x in range(w):
                # Get 3x3 neighborhood
                tl = self.get_pixel_safe(img, y - 1, x - 1)
                t = self.get_pixel_safe(img, y - 1, x)
                tr = self.get_pixel_safe(img, y - 1, x + 1)
                l = self.get_pixel_safe(img, y, x - 1)
                c = img[y, x].copy()  # center
                r = self.get_pixel_safe(img, y, x + 1)
                bl = self.get_pixel_safe(img, y + 1, x - 1)
                b = self.get_pixel_safe(img, y + 1, x)
                br = self.get_pixel_safe(img, y + 1, x + 1)
                
                # Replace None with center
                if tl is None: tl = c.copy()
                if t is None: t = c.copy()
                if tr is None: tr = c.copy()
                if l is None: l = c.copy()
                if r is None: r = c.copy()
                if bl is None: bl = c.copy()
                if b is None: b = c.copy()
                if br is None: br = c.copy()
                
                x2 = x * 2
                y2 = y * 2
                
                # Eagle algorithm rules
                # Top-left
                if self.pixels_equal(tl, t, threshold) and self.pixels_equal(tl, l, threshold):
                    out[y2, x2] = tl
                else:
                    out[y2, x2] = c
                
                # Top-right
                if self.pixels_equal(tr, t, threshold) and self.pixels_equal(tr, r, threshold):
                    out[y2, x2 + 1] = tr
                else:
                    out[y2, x2 + 1] = c
                
                # Bottom-left
                if self.pixels_equal(bl, b, threshold) and self.pixels_equal(bl, l, threshold):
                    out[y2 + 1, x2] = bl
                else:
                    out[y2 + 1, x2] = c
                
                # Bottom-right
                if self.pixels_equal(br, b, threshold) and self.pixels_equal(br, r, threshold):
                    out[y2 + 1, x2 + 1] = br
                else:
                    out[y2 + 1, x2 + 1] = c
        
        return out

    def nearest_neighbor_core(self, img, scale):
        """Nearest Neighbor scaling algorithm"""
        scale = int(round(scale))
        if scale <= 1:
            return img.copy()
        return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

    def _2xsai_core(self, img, threshold):
        """2xSaI algorithm implementation"""
        h, w = img.shape[:2]
        out = np.zeros((h * 2, w * 2, img.shape[2]), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                c = img[y, x].copy()
                
                # Get neighbors
                a = self.get_pixel_safe(img, y - 1, x)
                b = self.get_pixel_safe(img, y, x - 1)
                c_neighbor = self.get_pixel_safe(img, y, x + 1)
                d = self.get_pixel_safe(img, y + 1, x)
                a1 = self.get_pixel_safe(img, y - 1, x - 1)
                c1 = self.get_pixel_safe(img, y - 1, x + 1)
                d1 = self.get_pixel_safe(img, y + 1, x - 1)
                d2 = self.get_pixel_safe(img, y + 1, x + 1)

                # Fill missing neighbors
                if a is None: a = c.copy()
                if b is None: b = c.copy()
                if c_neighbor is None: c_neighbor = c.copy()
                if d is None: d = c.copy()
                if a1 is None: a1 = c.copy()
                if c1 is None: c1 = c.copy()
                if d1 is None: d1 = c.copy()
                if d2 is None: d2 = c.copy()

                # Default to center pixel
                out[y*2, x*2] = c
                out[y*2, x*2+1] = c
                out[y*2+1, x*2] = c
                out[y*2+1, x*2+1] = c

                # Interpolation
                if self.pixels_equal(a, b, threshold) and not self.pixels_equal(a, c_neighbor, threshold) and not self.pixels_equal(b, d, threshold):
                    out[y*2, x*2] = (a + b) / 2
                elif self.pixels_equal(a, c_neighbor, threshold) and not self.pixels_equal(a, b, threshold) and not self.pixels_equal(c_neighbor, d, threshold):
                    out[y*2, x*2+1] = (a + c_neighbor) / 2
                elif self.pixels_equal(b, d, threshold) and not self.pixels_equal(b, a, threshold) and not self.pixels_equal(d, c_neighbor, threshold):
                    out[y*2+1, x*2] = (b + d) / 2
                elif self.pixels_equal(c_neighbor, d, threshold) and not self.pixels_equal(c_neighbor, a, threshold) and not self.pixels_equal(d, b, threshold):
                    out[y*2+1, x*2+1] = (c_neighbor + d) / 2

        return out

    def super_2xsai_core(self, img, threshold):
        """Super 2xSaI algorithm implementation"""
        h, w = img.shape[:2]
        out = np.zeros((h * 2, w * 2, img.shape[2]), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                c = img[y, x].copy()
                
                # Get neighbors
                a = self.get_pixel_safe(img, y - 1, x)
                b = self.get_pixel_safe(img, y, x - 1)
                c_neighbor = self.get_pixel_safe(img, y, x + 1)
                d = self.get_pixel_safe(img, y + 1, x)
                a1 = self.get_pixel_safe(img, y - 1, x - 1)
                c1 = self.get_pixel_safe(img, y - 1, x + 1)
                d1 = self.get_pixel_safe(img, y + 1, x - 1)
                d2 = self.get_pixel_safe(img, y + 1, x + 1)

                # Fill missing neighbors
                if a is None: a = c.copy()
                if b is None: b = c.copy()
                if c_neighbor is None: c_neighbor = c.copy()
                if d is None: d = c.copy()
                if a1 is None: a1 = c.copy()
                if c1 is None: c1 = c.copy()
                if d1 is None: d1 = c.copy()
                if d2 is None: d2 = c.copy()

                # Default to center pixel
                out[y*2, x*2] = c
                out[y*2, x*2+1] = c
                out[y*2+1, x*2] = c
                out[y*2+1, x*2+1] = c

                # Super 2xSaI rules
                if self.pixels_equal(a, b, threshold) and not self.pixels_equal(a, c_neighbor, threshold) and not self.pixels_equal(b, d, threshold):
                    out[y*2, x*2] = a if self.color_diff(a, c) < self.color_diff(b, c) else b
                elif self.pixels_equal(a, c_neighbor, threshold) and not self.pixels_equal(a, b, threshold) and not self.pixels_equal(c_neighbor, d, threshold):
                    out[y*2, x*2+1] = a if self.color_diff(a, c) < self.color_diff(c_neighbor, c) else c_neighbor
                elif self.pixels_equal(b, d, threshold) and not self.pixels_equal(b, a, threshold) and not self.pixels_equal(d, c_neighbor, threshold):
                    out[y*2+1, x*2] = b if self.color_diff(b, c) < self.color_diff(d, c) else d
                elif self.pixels_equal(c_neighbor, d, threshold) and not self.pixels_equal(c_neighbor, a, threshold) and not self.pixels_equal(d, b, threshold):
                    out[y*2+1, x*2+1] = c_neighbor if self.color_diff(c_neighbor, c) < self.color_diff(d, c) else d

        return out

    def super_eagle_core(self, img, threshold):
        """Super Eagle algorithm implementation"""
        h, w = img.shape[:2]
        out = np.zeros((h * 2, w * 2, img.shape[2]), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                c = img[y, x].copy()
                
                # Get neighbors
                tl = self.get_pixel_safe(img, y - 1, x - 1)
                t = self.get_pixel_safe(img, y - 1, x)
                tr = self.get_pixel_safe(img, y - 1, x + 1)
                l = self.get_pixel_safe(img, y, x - 1)
                r = self.get_pixel_safe(img, y, x + 1)
                bl = self.get_pixel_safe(img, y + 1, x - 1)
                b = self.get_pixel_safe(img, y + 1, x)
                br = self.get_pixel_safe(img, y + 1, x + 1)

                # Fill missing neighbors
                if tl is None: tl = c.copy()
                if t is None: t = c.copy()
                if tr is None: tr = c.copy()
                if l is None: l = c.copy()
                if r is None: r = c.copy()
                if bl is None: bl = c.copy()
                if b is None: b = c.copy()
                if br is None: br = c.copy()

                # Default to center pixel
                out[y*2, x*2] = c
                out[y*2, x*2+1] = c
                out[y*2+1, x*2] = c
                out[y*2+1, x*2+1] = c
                
                # Super Eagle rules
                if self.pixels_equal(l, tl, threshold) and self.pixels_equal(t, tl, threshold):
                    out[y*2, x*2] = tl
                else:
                    out[y*2, x*2] = (c*2 + t + l)/4
                
                if self.pixels_equal(t, tr, threshold) and self.pixels_equal(r, tr, threshold):
                    out[y*2, x*2+1] = tr
                else:
                    out[y*2, x*2+1] = (c*2 + t + r)/4

                if self.pixels_equal(l, bl, threshold) and self.pixels_equal(b, bl, threshold):
                    out[y*2+1, x*2] = bl
                else:
                    out[y*2+1, x*2] = (c*2 + b + l)/4

                if self.pixels_equal(r, br, threshold) and self.pixels_equal(b, br, threshold):
                    out[y*2+1, x*2+1] = br
                else:
                    out[y*2+1, x*2+1] = (c*2 + b + r)/4

        return out
    
    def scale_pixel_art(self, image, algorithm, threshold):
        """Main scaling function"""
        # Convert from ComfyUI tensor format to numpy
        batch_numpy = image.cpu().numpy()
        
        # Process each image in the batch
        results = []
        
        for img in batch_numpy:
            # Ensure proper shape (H, W, C)
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
            
            # Make sure we have float32 for calculations
            img = img.astype(np.float32)
            
            # Apply selected algorithm
            if algorithm == "NearestNeighbor2x":
                scaled = self.nearest_neighbor_core(img, 2)
            elif algorithm == "NearestNeighbor3x":
                scaled = self.nearest_neighbor_core(img, 3)
            elif algorithm == "NearestNeighbor4x":
                scaled = self.nearest_neighbor_core(img, 4)
            elif algorithm == "Scale2x":
                scaled = self.scale2x_core(img, threshold)
            elif algorithm == "Scale3x":
                scaled = self.scale2x_core(img, threshold)
                scaled = self.scale2x_core(scaled, threshold)
                h, w = img.shape[:2]
                scaled = scaled[:h*3, :w*3]
            elif algorithm == "HQ2x":
                scaled = self.hq2x_core(img, threshold)
            elif algorithm == "HQ3x":
                scaled = self.hq2x_core(img, threshold)
                scaled = self.scale2x_core(scaled, threshold)
                h, w = img.shape[:2]
                scaled = scaled[:h*3, :w*3]
            elif algorithm == "HQ4x":
                scaled = self.hq2x_core(img, threshold)
                scaled = self.hq2x_core(scaled, threshold)
            elif algorithm == "xBR2x":
                scaled = self.xbr2x_core(img, threshold)
            elif algorithm == "xBR3x":
                scaled = self.xbr2x_core(img, threshold)
                scaled = self.scale2x_core(scaled, threshold)
                h, w = img.shape[:2]
                scaled = scaled[:h*3, :w*3]
            elif algorithm == "xBR4x":
                scaled = self.xbr2x_core(img, threshold)
                scaled = self.xbr2x_core(scaled, threshold)
            elif algorithm == "Eagle2x":
                scaled = self.eagle2x_core(img, threshold)
            elif algorithm == "Eagle3x":
                scaled = self.eagle2x_core(img, threshold)
                scaled = self.scale2x_core(scaled, threshold)
                h, w = img.shape[:2]
                scaled = scaled[:h*3, :w*3]
            elif algorithm == "Eagle4x":
                scaled = self.eagle2x_core(img, threshold)
                scaled = self.eagle2x_core(scaled, threshold)
            elif algorithm == "2xSaI":
                scaled = self._2xsai_core(img, threshold)
            elif algorithm == "Super2xSaI":
                scaled = self.super_2xsai_core(img, threshold)
            elif algorithm == "SuperEagle":
                scaled = self.super_eagle_core(img, threshold)
            else:
                scaled = img
            
            # Ensure output is in [0, 1] range
            scaled = np.clip(scaled, 0.0, 1.0)
            
            results.append(scaled)
        
        # Convert back to tensor
        output = torch.from_numpy(np.array(results)).float()
        
        return (output,)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "PixelArtScaler": PixelArtScaler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelArtScaler": "Pixel Art Scaler (HQx/xBR)"
}