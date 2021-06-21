import numpy as np

def find_connected_components(branch_points, peak_points, imsz) :
    connected_components = {}
    size_component = []
    num_connected_component = 0
    tag_which_component = np.zeros(imsz)

    tag_is_branch = np.zeros(imsz)

    for i in range(branch_points.shape[0]):
        x = int(branch_points[i, 0])
        y = int(branch_points[i, 1])
        tag_is_branch[y-1, x-1] = 1

    for i in range(branch_points.shape[0]):
        x = int(branch_points[i, 0])
        y = int(branch_points[i, 1])
        index_component = -1
        if tag_which_component[y-1, x-1] == 0:
            flag_found_one_neighbor = False
            for m in range(-1, 2, 1):
                for n in range(-1, 2, 1):
                    if tag_is_branch[y+m-1, x+n-1] == 1 and tag_which_component[y+m-1, x+n-1] >0:
                        index_component = int(tag_which_component[y+m-1, x+n-1])
                        flag_found_one_neighbor = True
                        break

            if flag_found_one_neighbor == False:
                num_connected_component += 1
                index_component = num_connected_component
                size_component.append(0)
        else:
            index_component = int(tag_which_component[y-1, x-1])

        for m in range(-1, 2, 1):
            for n in range(-1, 2, 1):
                if tag_is_branch[y+m-1, x+n-1] == 1 and tag_which_component[y+m-1, x+n-1] <= 0:
                    tag_which_component[y+m-1, x+n-1] = index_component
                    connected_components_temp = np.array([x+n, y+m])
                    connected_components[index_component-1] = np.vstack((connected_components[index_component-1], connected_components_temp)) if index_component-1 in connected_components.keys() else connected_components_temp
                    size_component[index_component-1] = size_component[index_component-1]+1
    pair_merge_component = np.array([])
    num_pair = 0

    for i in range(0, branch_points.shape[0]):
        x = int(branch_points[i, 0])
        y = int(branch_points[i, 1])
        for m in range(-1, 2, 1):
            for n in range(-1, 2, 1):
                if tag_which_component[y+m-1, x+n-1] != tag_which_component[y-1, x-1] and tag_which_component[y-1, x-1] >0 and tag_which_component[y+m-1, x+n-1] >0:
                    num_pair += 1
                    pair_merge_component_temp = np.array([min(tag_which_component[y-1, x-1], tag_which_component[y+m-1, x+n-1]), max(tag_which_component[y-1, x-1], tag_which_component[y+m-1, x+n-1])])
                    pair_merge_component = np.vstack((pair_merge_component, pair_merge_component_temp)) if pair_merge_component.size != 0 else pair_merge_component_temp
    equivalent_component = np.zeros((num_connected_component, 1))

    for i in range(num_pair):
        a = int(pair_merge_component[i, 0])
        b = int(pair_merge_component[i, 1])

        if equivalent_component[a-1] == 0:
            equivalent_component[b-1] = a
        else:
            equivalent_component[b-1] = equivalent_component[a-1]

    table_component_index_old2new = np.zeros((num_connected_component, 1))
    previous = 0

    for i in range(num_connected_component):
        if equivalent_component[i] == 0:
            table_component_index_old2new[i] = previous +1
            previous+= 1
        else:
            table_component_index_old2new[i] = table_component_index_old2new[int(equivalent_component[i])-1]

    num_connected_component = num_connected_component-np.sum(equivalent_component>0)
    connected_component_final = {}
    size_component_final = np.zeros((num_connected_component, 1))
    connected_peaks = {}
    for i in range(branch_points.shape[0]):
        x = int(branch_points[i, 0])
        y = int(branch_points[i, 1])
        if tag_which_component[y-1, x-1] >0:
            if equivalent_component[int(tag_which_component[y-1, x-1])-1] > 0:
                tag_which_component[y-1, x-1] = equivalent_component[int(tag_which_component[y-1, x-1])-1]

            new_index = int(table_component_index_old2new[int(tag_which_component[y-1, x-1])-1])
            tag_which_component[y-1, x-1] = new_index
            size_component_final[new_index-1] = size_component_final[new_index-1]+1
            connected_component_final_temp = np.array([x, y])
            connected_component_final[new_index-1] = np.vstack((connected_component_final[new_index-1], connected_component_final_temp)) if new_index-1 in connected_component_final.keys() else connected_component_final_temp
            k = int(size_component_final[new_index-1])

            connected_peaks[new_index-1] = np.vstack((connected_peaks[new_index-1], peak_points[2*i, :])) if new_index-1 in connected_peaks.keys() else peak_points[2*i, :]
            connected_peaks[new_index-1] = np.vstack((connected_peaks[new_index-1], peak_points[2*i+1, :]))

    return connected_component_final, tag_which_component, connected_peaks
