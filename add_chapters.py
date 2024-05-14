# import re

# # Read the markdown content
# with open('Thesis.md', 'r') as file:
#     content = file.read()

# # Regex to find chapters and sections
# # These regexes accurately capture the appropriate markdown headings.
# chapter_regex = re.compile(r'^## (?!\s*#)(.*)', re.MULTILINE)
# section_regex = re.compile(r'^### (?!\s*#)(.*)', re.MULTILINE)

# # Variables to hold current chapter and section numbers
# chapter_number = 0
# section_number = 0

# # Function to replace chapter headings and reset section number
# def replace_chapter(match):
#     global chapter_number, section_number
#     chapter_number += 1
#     section_number = 0  # Reset section number at the start of a new chapter
#     return f"## Chapter {chapter_number}. {match.group(1)}"

# # Function to replace section headings for each chapter
# def replace_section(match):
#     global section_number
#     section_number += 1
#     return f"### Section {section_number}. {match.group(1)}"

# # Apply the regex to replace chapter and then section headings
# content = chapter_regex.sub(replace_chapter, content)
# content = section_regex.sub(replace_section, content)

# # Write the updated content back to the file or a new file
# with open('Updated_Thesis.md', 'w') as file:
#     file.write(content)
import re

import re
import os
import qrcode

def create_qr_code(output_path):
    print("Creating QR codes for video links in the markdown file...")
    # Define the input markdown file and output directory for QR codes
    output_directory = "qr_codes"
    input_markdown_file = output_path
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Read the markdown file
    with open(input_markdown_file, "r") as file:
        content = file.read()

    # Regex pattern to find video links
    video_link_pattern = re.compile(r"\[.*\]\(https:\/\/github.com\/driessenslucas\/researchproject\/assets\/[^\s)]+\)")


    # Find all video links
    video_links = video_link_pattern.findall(content)

    # get the names out of the links the names are in [name](link) format
    video_names = [re.search(r"\[(.*)\]", link).group(1) for link in video_links]

    #remove the names from the links
    video_links = [re.sub(r"\[.*\]", "", link) for link in video_links]
    
    # Generate QR codes for each video link and save them
    qr_code_files = []
    for i, link in enumerate(video_links):
        # Generate QR code
        img = qrcode.make(link)
        qr_code_filename = os.path.join(output_directory, f"qr_code_{i+1}.png")
        img.save(qr_code_filename)
        qr_code_files.append(qr_code_filename)

    print("QR codes have been generated for video links.")
    print(qr_code_files)

    # Update the markdown content with QR code images
    for i, link in enumerate(video_links):
        print(f"Adding QR code for: {link}")
        # qr_code_image_tag = f"   - ![QR code for video {video_names[i]}](./{qr_code_files[i]})" + "{width=10% height=10%}"
        qr_code_image_tag = f'''
    \\begin{{figure}}[h]
        \\centering
        \\begin{{minipage}}{{0.2\\textwidth}}
            \\includegraphics[width=1in]{{{qr_code_files[i]}}}
        \\end{{minipage}}
        \\caption{{QR code for video {video_names[i]}. (Video by author.)}}
    \\end{{figure}}
'''
        content = content.replace(link, f"{link}\n{qr_code_image_tag}")

    # Save the updated content back to the markdown file
    with open(input_markdown_file, "w") as file:
        file.write(content)

    print("Markdown file has been updated with QR code images.")

def process_markdown_file(input_path, output_path):
    # Initialize variables to keep track of chapters and sections
    chapter_number = 0
    processed_content = []

    # Open and read the file
    with open(input_path, "r") as file:
        lines = file.readlines()
    lines_to_ignore = [
        "## Glossary of Terms",
        "## List of Abbreviations",
        "## References",
    ]
    # Process each line
    for line in lines:
        # Check for chapter headings
        if line.startswith("## ") and line.strip() not in lines_to_ignore:
            chapter_number += 1
            section_number = 0  # Reset section number for new chapter
            chapter_title = line.strip("# ").strip()
            processed_line = f"## Chapter {chapter_number}. {chapter_title}\n"
        # Check for section headings
        elif (
            line.startswith("### ")
            and chapter_number > 0
            and line.strip() not in lines_to_ignore
        ):  # Ensure we're within a chapter
            section_number += 1
            subsection_number = 0  # Reset subsection number for new section
            section_title = line.strip("# ").strip()
            processed_line = f"### {chapter_number}.{section_number}. {section_title}\n"
        elif (
            line.startswith("#### ")
            and chapter_number > 0
            and line.strip() not in lines_to_ignore
        ):
            subsection_number += 1
            sub_subsection_number = 0  # Reset sub-subsection number for new subsection
            sub_section_title = line.strip("# ").strip()
            processed_line = f"#### {chapter_number}.{section_number}.{subsection_number}. {sub_section_title}\n"
        # elif line.startswith("##### ") and chapter_number > 0 and line.strip() not in lines_to_ignore:
        #     sub_sub_section_title = line.strip('# ').strip()
        #     sub_subsection_number += 1
        #     processed_line = f"##### {chapter_number}.{section_number}.{subsection_number}.{sub_subsection_number}. {sub_sub_section_title}\n"
        else:
            processed_line = line

        # Append processed or original line to the content list
        processed_content.append(processed_line)

    # Write the processed content back to a new file
    with open(output_path, "w") as file:
        file.writelines(processed_content)

    create_qr_code(output_path)

# Usage example:
process_markdown_file("Thesis.md", "Updated_Thesis.md")
