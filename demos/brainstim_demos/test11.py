import tkinter as tk
from tkinter import ttk

def function():
    def submit():
        paradigm = paradigm_var.get()
        rows = int(rows_entry.get())
        columns = int(columns_entry.get())
        n_elements = int(n_elements_entry.get())
        stim_length = int(stim_length_entry.get())
        stim_width = int(stim_width_entry.get())
        display_time = int(display_time_entry.get())
        index_time = int(index_time_entry.get())
        response_time = int(response_time_entry.get())
        nrep = int(nrep_entry.get())
        online = online_var.get()
        device_type = device_type_var.get()

        print("Values submitted:")
        print(f"Paradigm: {paradigm}")
        print(f"Rows: {rows}")
        print(f"Columns: {columns}")
        print(f"Number of Elements: {n_elements}")
        print(f"Stimulus Length: {stim_length}")
        print(f"Stimulus Width: {stim_width}")
        print(f"Display Time: {display_time}")
        print(f"Index Time: {index_time}")
        print(f"Response Time: {response_time}")
        print(f"Number of Repetitions: {nrep}")
        print(f"Online: {online}")
        print(f"Device Type: {device_type}")

        parameters = {
            'paradigm': paradigm,
            'rows': rows,
            'columns': columns,
            'n_elements': n_elements,
            'stim_length': stim_length,
            'stim_width': stim_width,
            'display_time': display_time,
            'index_time': index_time,
            'response_time': response_time,
            'nrep': nrep,
            'online': online,
            'device_type': device_type
        }

        root.destroy()


    # Create a tkinter window
    root = tk.Tk()
    root.attributes("-alpha", 0.85)
    root.title("Experiment Settings")

    # Create labels and entries for each parameter
    parameters = ['Paradigm', 'Rows', 'Columns', 'Number of Elements', 'Stimulus Length', 'Stimulus Width',
                  'Display Time', 'Index Time', 'Response Time', 'Number of Repetitions', 'Online', 'Device']

    default_values = ['4', '5', '20', '200', '200', '1', '1', '2', '5']

    for i, parameter in enumerate(parameters):
        ttk.Label(root, text=parameter).grid(row=i, column=0)

    paradigm_var = tk.StringVar()
    paradigm_combobox = ttk.Combobox(root, textvariable=paradigm_var, values=['SSVEP',])
    paradigm_combobox.current(0)
    paradigm_combobox.grid(row=0, column=1)

    # Insert default values to the Entry widgets
    rows_entry = ttk.Entry(root)
    rows_entry.insert(0, default_values[0])  # Insert default value
    rows_entry.grid(row=1, column=1)

    columns_entry = ttk.Entry(root)
    columns_entry.insert(0, default_values[1])  # Insert default value
    columns_entry.grid(row=2, column=1)

    n_elements_entry = ttk.Entry(root)
    n_elements_entry.insert(0, default_values[2])  # Insert default value
    n_elements_entry.grid(row=3, column=1)

    stim_length_entry = ttk.Entry(root)
    stim_length_entry.insert(0, default_values[3])  # Insert default value
    stim_length_entry.grid(row=4, column=1)

    stim_width_entry = ttk.Entry(root)
    stim_width_entry.insert(0, default_values[4])  # Insert default value
    stim_width_entry.grid(row=5, column=1)

    display_time_entry = ttk.Entry(root)
    display_time_entry.insert(0, default_values[5])  # Insert default value
    display_time_entry.grid(row=6, column=1)

    index_time_entry = ttk.Entry(root)
    index_time_entry.insert(0, default_values[6])  # Insert default value
    index_time_entry.grid(row=7, column=1)

    response_time_entry = ttk.Entry(root)
    response_time_entry.insert(0, default_values[7])  # Insert default value
    response_time_entry.grid(row=8, column=1)

    nrep_entry = ttk.Entry(root)
    nrep_entry.insert(0, default_values[8])  # Insert default value
    nrep_entry.grid(row=9, column=1)

    online_var = tk.StringVar()
    online_combobox = ttk.Combobox(root, textvariable=online_var, values=['Yes', 'No'])
    online_combobox.current(1)
    online_combobox.grid(row=10, column=1)

    device_type_var = tk.StringVar()
    device_type_combobox = ttk.Combobox(root, textvariable=device_type_var, values=['BlueBCI', 'NeuroScan', 'Neuracle',])
    device_type_combobox.current(0)
    device_type_combobox.grid(row=11, column=1)

    # Submit button
    submit_button = ttk.Button(root, text="Submit", command=submit, style='Custom.TButton')
    submit_button.grid(row=12, column=0, columnspan=2, padx=10, pady=10, sticky='we')

    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)
    root.mainloop()

if __name__ == "__main__":
    function()
