# Task 24b notes: pixel size metadata check (read-only)

All 8 stores (4 cores x H&E, COMET) have `OME/METADATA.ome.xml` (Bio-Formats 8.2.0 writer). Units are micrometers in XML. The XML has no objective, magnification, instrument or scanner fields (only Pixels, Channel and a pyramid-size annotation), so there is nothing scanner-related to report.

| Core, modality | XML PhysicalSizeX = Y (um) | multiscales level 0 (um) | XML vs multiscales | Implied by provided alignment (um) | Implied vs multiscales |
|---|---|---|---|---|---|
| all 4 cores, H&E | 0.203125 | 0.203125 | 0.00% | 0.19999 to 0.20031 | -1.54% to -1.39% |
| core01 COMET | **0.2125** | 0.28 | **-24.1%** | 0.27922 | -0.28% |
| core02, 03, 04 COMET | 0.28 | 0.28 | 0.00% | 0.27908 to 0.27919 | -0.33% to -0.29% |

Agreement:
- H&E: XML and multiscales agree exactly. The alignment-implied size (about 0.2001 um) disagrees with both by about 1.4% to 1.5%. Cause not determined.
- COMET cores 02 to 04: XML and multiscales agree exactly; the alignment-implied size is 0.3% lower than both.
- COMET core01: XML (0.2125) disagrees with multiscales (0.28) by 24%. The alignment-implied size (0.2792) agrees with the multiscales value, not the XML, so the core01 COMET XML value appears wrong. 0.2125 is also the Xenium Explorer pixel size; why it is in this file is not known. The app reads multiscales (0.28), so it is not affected.
