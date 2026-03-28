```markdown
# Design System Document: Forensic Precision & Editorial Authority

## 1. Overview & Creative North Star: "The Digital Pathologist"
This design system moves away from the "SaaS-dashboard" trope of blue boxes and generic grids. Our North Star is **The Digital Pathologist**. It balances the cold, absolute precision of forensic science with the high-end editorial authority of a premium broadsheet. 

To achieve an Apple Pro Display XDR aesthetic, we utilize a "High-Contrast Minimalist" approach. We break the template look through **Intentional Asymmetry**: placing heavy, bold serif metrics against ultra-fine, technical sans-serif metadata. The UI should feel like a backlit glass specimen table in a darkened laboratory—expensive, focused, and indisputable.

---

## 2. Colors: Obsidian & Crimson
The palette is rooted in an absolute dark mode to maximize the perceived contrast of the crimson highlights.

### Surface Hierarchy & Nesting
We reject the flat UI. We build depth through **Tonal Layering**.
*   **Base Layer:** `surface_container_lowest` (#0e0e0e) or `background` (#131313) for the main canvas.
*   **Secondary Layer:** Use `surface_container_low` (#1c1b1b) for sidebar or navigation regions.
*   **The Component Layer:** Use `surface_container` (#201f1f) with a `backdrop-filter: blur(20px)` for primary workspace cards.

### The "No-Line" Rule
Standard 1px solid borders are strictly prohibited for sectioning. Structural boundaries must be defined by:
1.  **Background Color Shifts:** A `surface_container_high` card sitting on a `surface` background.
2.  **Negative Space:** Utilizing the `spacing-10` or `spacing-12` tokens to create "void-based" separation.

### The "Glass & Gradient" Rule
For elevated forensic tools or floating inspectors, use a **Glassmorphism** stack:
*   Background: `surface_variant` at 40% opacity.
*   Blur: `backdrop-filter: blur(12px)`.
*   Border: `outline_variant` at 10% opacity (The "Ghost Border").

---

## 3. Typography: The Tension of Two Worlds
We use a high-contrast pairing of **Newsreader (Serif)** and **Inter (Sans-Serif)** to convey both history and technology.

*   **Display & Headline (Newsreader):** Use for "Key Evidence" metrics and section headers. The bold crimson-red (`primary`) serif evokes a sense of "The Smoking Gun"—authoritative and serious. 
    *   *Example:* `display-lg` in `#ffb3b1` for a document's authenticity score.
*   **Title & Body (Inter):** Use for technical metadata, file paths, and logs. This provides the "High-Precision" feel of a laboratory report.
    *   *Example:* `body-sm` in `on_surface_variant` for timestamps and SHA-256 hashes.

---

## 4. Elevation & Depth: Tonal Sculpting
We do not use drop shadows to show importance; we use light.

*   **The Layering Principle:** To "lift" a document preview, place a `surface_container_highest` (#353534) element inside a `surface_container_lowest` (#0e0e0e) container. The shift in value creates natural prominence.
*   **Ambient Shadows:** For floating modals, use a "Forensic Glow." Instead of black shadows, use `primary_container` at 5% opacity with a `100px` blur. It should feel like light escaping from behind the element.
*   **Ghost Borders:** Where visual containment is required for high-density data, use the `px` spacing token with `outline_variant` (#5b403d) set to 15% opacity. This creates a "hairline" effect reminiscent of high-end optical equipment.

---

## 5. Components

### Primary Metrics (The Crimson Hero)
*   **Styling:** `display-md` Newsreader text in `primary`.
*   **Layout:** Offset to the left with a minimalist `secondary` progress bar (2px height) running underneath.

### Buttons: The "Tactile Precision" Set
*   **Primary:** Solid `primary_container` with `on_primary_container` text. Border-radius: `sm` (0.125rem) for a sharp, technical look.
*   **Tertiary (Technical):** Ghost style. No background, `outline_variant` at 20% opacity border, `label-md` Inter text. Use for low-priority forensic actions.

### Minimalist Red Progress Bars
*   **Track:** `surface_container_highest` at 100% width.
*   **Indicator:** 2px height, `primary` color. No rounded caps; use `none` or `sm` radius for a clinical finish.

### Forensic Data Lists
*   **The "No-Divider" Rule:** Forbid 1px dividers between list items. Use a background shift to `surface_container_low` on hover or increase the vertical spacing to `spacing-4`.
*   **Metadata Labels:** Use `label-sm` in `on_tertiary_fixed_variant` for a subtle blue-tinted tech feel against the obsidian background.

### Document "Glass" Cards
*   **Base:** `surface_container` at 60% opacity.
*   **Border:** Top and left sides only using `outline_variant` at 10% to mimic a "rim light" effect on glass.

---

## 6. Do's and Don'ts

### Do:
*   **Embrace the Void:** Let the `#0a0a0a` background breathe. Large margins (`spacing-16`+) create a premium, gallery-like feel.
*   **Use Red Sparingly:** Crimson (`primary`) should only represent critical forensic findings or "Hot" data points.
*   **Precision Alignment:** Use the `0.5` and `1` spacing tokens to align technical text with sub-pixel precision.

### Don't:
*   **Don't use Rounded Corners:** Avoid `xl` or `full` rounding. Stick to `none`, `sm`, or `md`. We want the UI to feel like a sharp, machined instrument, not a consumer social app.
*   **Don't use Pure White:** Avoid `#ffffff` for text. Use `on_surface` (#e5e2e1) to prevent eye strain against the obsidian background and maintain the "Pro Display" tonal range.
*   **Don't use Standard Shadows:** Never use high-opacity, small-blur shadows. It breaks the glass aesthetic. Use tonal shifts or ambient glows instead.

### Interaction Note
When a user hovers over a forensic element, the "Ghost Border" should transition from 10% to 40% opacity, and the `backdrop-filter` should increase. This mimics the "focusing" of a microscope lens.```