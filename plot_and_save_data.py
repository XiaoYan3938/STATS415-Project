import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from matplotlib import cm
import numpy as np
import pandas as pd
import csv
from datetime import timedelta
import seaborn as sns


def read_csv_data(filename):
    data = {}
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["comments_disabled"] != "True" and row["ratings_disabled"] != "True" and row["video_error_or_removed"] != "True":
                if row["video_id"] not in data:
                    data[row["video_id"]] = {"category_id": row["category_id"],
                                                            "publish_time": row["publish_time"],
                                                            "appear_times": 0
                                                            }
                    data[row["video_id"]]["trending_date"] = []
                    data[row["video_id"]]["views"] = []
                    data[row["video_id"]]["likes"] = []
                    data[row["video_id"]]["dislikes"] = []
                    data[row["video_id"]]["comments"] = []
                # skip duplicated data (actually there are some)
                if row["trending_date"] in data[row["video_id"]]["trending_date"]:
                    continue
                data[row["video_id"]]["trending_date"].append(row["trending_date"])
                data[row["video_id"]]["views"].append(row["views"])
                data[row["video_id"]]["likes"].append(row["likes"])
                data[row["video_id"]]["dislikes"].append(row["dislikes"])
                data[row["video_id"]]["comments"].append(row["comment_count"])
                data[row["video_id"]]["appear_times"] += 1
    return data


def get_and_plot_vldca(data, plot=False):
    views_data = []
    likes_data = []
    dislikes_data = []
    comments_data = []
    appear_times = []
    for video in data:
        views_data += [float(x)/1000000 for x in data[video]["views"]]
        likes_data += [float(x)/1000000 for x in data[video]["likes"]]
        dislikes_data += [float(x)/1000000 for x in data[video]["dislikes"]]
        comments_data += [float(x)/1000000 for x in data[video]["comments"]]
        appear_times.append(data[video]["appear_times"])
    if plot:
        plt.figure()
        plt.subplot(331)
        plt.scatter(likes_data, views_data, s=1, c="blue")
        plt.xlabel("likes(M)")
        plt.ylabel("views(M)")
        plt.subplot(332)
        plt.scatter(dislikes_data, views_data, s=1, c="blue")
        plt.subplot(333)
        plt.scatter(comments_data, views_data, s=1, c="blue")
        plt.subplot(335)
        plt.scatter(dislikes_data, likes_data, s=1, c="blue")
        plt.xlabel("dislikes(M)")
        plt.ylabel("likes(M)")
        plt.subplot(336)
        plt.scatter(comments_data, likes_data, s=1, c="blue")
        plt.subplot(339)
        plt.scatter(comments_data, dislikes_data, s=1, c="blue")
        plt.xlabel("comments(M)")
        plt.ylabel("dislikes(M)")
        savefig("./figures/US_video_data_relationship.png")
        plt.close()

        labels = ["", "views", "", "likes", "", "dislikes", "", "comments"]
        total = np.array([views_data, likes_data, dislikes_data, comments_data])
        corr_matrix = np.corrcoef(total)
        fig, ax = plt.subplots()
        heatmap = ax.imshow(corr_matrix, interpolation="nearest", cmap=cm.coolwarm)
        cbar_min = corr_matrix.min().min()
        cbar_max = corr_matrix.max().max()
        cbar = fig.colorbar(heatmap, ticks=[cbar_min, cbar_max])
        ax.set_yticklabels(labels)
        ax.set_xticklabels(labels)
        axis = [0, 1, 2, 3]
        for m in axis:
            for n in axis:
                if m >= n:
                    plt.text(m, n, round(corr_matrix[m][n], 3), {"color": "black", "fontsize": 20}, ha="center", va="center")
        savefig("./figures/US_video_data_correlation_coefficient.png")
        plt.close()

        views_latest = []
        likes_latest = []
        dislikes_latest = []
        comments_latest = []
        for video in data:
            views_latest.append(float(data[video]["views"][-1])/1000000)
            likes_latest.append(float(data[video]["likes"][-1])/1000000)
            dislikes_latest.append(float(data[video]["dislikes"][-1])/1000000)
            comments_latest.append(float(data[video]["comments"][-1])/1000000)
        plt.figure(figsize=(10,8))
        plt.subplot(221)
        plt.scatter(views_latest, appear_times, s=1, c="blue")
        plt.xlabel("views(M)")
        plt.ylabel("trend duration")
        plt.subplot(222)
        plt.scatter(likes_latest, appear_times, s=1, c="blue")
        plt.xlabel("likes(M)")
        plt.subplot(223)
        plt.scatter(dislikes_latest, appear_times, s=1, c="blue")
        plt.xlabel("dislikes(M)")
        plt.subplot(224)
        plt.scatter(comments_latest, appear_times, s=1, c="blue")
        plt.xlabel("comments(M)")
        savefig("./figures/US_video_trend_duration.png")
        plt.close()
    return views_data, likes_data, dislikes_data, comments_data, appear_times


def plot_number_and_trend_duration(data):
    appearances = {}
    for video in data:
        if data[video]["appear_times"] not in appearances:
            appearances[data[video]["appear_times"]] = 0
        appearances[data[video]["appear_times"]] += 1
    appearances_sorted = sorted(appearances.items(), key=lambda k:k[0])
    x_appearance = []
    y_appearance = []
    for item in appearances_sorted:
        x_appearance.append(item[0])
        y_appearance.append(item[1])
    plt.plot(x_appearance, y_appearance, "b")
    plt.xlabel("trend duration")
    plt.xticks(np.linspace(0, 35, 8))
    plt.ylabel("video number")
    savefig("./figures/US_video_trend_length.png")
    plt.close()


def plot_sample_log_view_percentage_change_for_a_video(data, times_limit):
    for video in data:
        trending_dates = []
        for trending_date in data[video]["trending_date"]:
            trending_dates.append(pd.to_datetime(trending_date, errors="coerce", format="%y.%d.%m").date())
        dif_trending_dates = []
        for n in range(len(trending_dates)-1):
            dif_trending_dates.append((trending_dates[n+1]-trending_dates[n])/timedelta(days=1))
            if dif_trending_dates[n] == 0:
                dif_trending_dates[n] += 1
        times = data[video]["appear_times"]
        if times >= times_limit:
            views = data[video]["views"]
            log_dif_views_ratio = []
            for n in range(len(views)-1):
                log_dif_views_ratio.append(np.log((int(views[n+1])-int(views[n]))/dif_trending_dates[n]/int(views[n])))
            plt.plot(range(len(log_dif_views_ratio)), log_dif_views_ratio, "b.")
            plt.xlabel("time")
            plt.ylabel("log views percentage change per day")
            savefig("./figures/US_sample_video_log_views_per_day_percentage_change.png")
            plt.close()
            break


def add_publish_to_trend(data):
    publish_to_trend = []
    for video in data:
        first_trending_date = pd.to_datetime(data[video]["trending_date"][0], errors="coerce", format="%y.%d.%m").date()
        publish_date = pd.to_datetime(data[video]["publish_time"], errors="coerce", format="%Y-%m-%dT%H:%M:%S.%fZ").date()
        publish_to_trend_single = (first_trending_date - publish_date) / timedelta(days=1)
        data[video]["publish_to_trend"] = publish_to_trend_single
        publish_to_trend.append(publish_to_trend_single)
    return publish_to_trend


def plot_publish_to_trend_heatmap(data):
    appear_times = get_and_plot_vldca(data)[-1]
    publish_to_trend = add_publish_to_trend(data)
    video_same_count = {}
    for item1, item2 in zip(appear_times, publish_to_trend):
        if str(item1)+"|"+str(item2) not in video_same_count:
            video_same_count[str(item1)+"|"+str(item2)] = 0
        video_same_count[str(item1)+"|"+str(item2)] += 1
    combine_data = []
    for item in video_same_count:
        combine_data.append([float(item.split("|")[0]), float(item.split("|")[1]), video_same_count[item]])
    heat_map_data = np.zeros((40, 40))
    for i in range(40):
        for j in range(40):
            for row in combine_data:
                if row[0] == i+1 and row[1] == j:
                    heat_map_data[i][j] = row[2]
    plt.figure(figsize=(15,15))
    ax = sns.heatmap(heat_map_data, mask=heat_map_data == 0, cmap="plasma_r")
    ax.invert_yaxis()
    plt.xlabel("publish to trend", fontsize=30)
    plt.ylabel("trend duration", fontsize=30)
    savefig("./figures/heatmap_for_publish_to_trend_and_trend_duration.png")
    plt.close()


def save_data_for_SVC(data, filename):
    with open(filename, "w") as f:
        field_names = ["will_end_trend_at_period_t+1_or_not", "t_period_log_views_per_day_percentage_change", "appeared_times"]
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for video in data:
            trending_dates = []
            for trending_date in data[video]["trending_date"]:
                trending_dates.append(pd.to_datetime(trending_date, errors="coerce", format="%y.%d.%m").date())
            dif_trending_dates = []
            for n in range(len(trending_dates)-1):
                dif_trending_dates.append((trending_dates[n+1]-trending_dates[n])/timedelta(days=1))
                if dif_trending_dates[n] == 0:
                    dif_trending_dates[n] += 1
            times = data[video]["appear_times"]
            if times >= 2:
                views = data[video]["views"]
                dif_views_ratio = []
                for n in range(len(views)-1):
                    if int(views[n+1])-int(views[n]) > 0:
                        dif_views_ratio.append(np.log((int(views[n+1])-int(views[n]))/dif_trending_dates[n]/int(views[n])))
                    if int(views[n+1])-int(views[n]) <= 0:
                        dif_views_ratio.append("na")
            for n in range(len(dif_views_ratio)):
                if dif_views_ratio[n] != "na":
                    writer.writerow({"will_end_trend_at_period_t+1_or_not": n == (len(dif_views_ratio) - 1),
                                     "t_period_log_views_per_day_percentage_change": dif_views_ratio[n],
                                     "appeared_times": n+1})
    

if __name__ == "__main__":
    us_data = read_csv_data("USvideos.csv")
    get_and_plot_vldca(us_data, plot=True)
    plot_number_and_trend_duration(us_data)
    plot_sample_log_view_percentage_change_for_a_video(us_data, 20)
    plot_publish_to_trend_heatmap(us_data)
    save_data_for_SVC(us_data, "US_videos_SVC.csv")
    gb_data = read_csv_data("GBvideos.csv")
    save_data_for_SVC(gb_data, "GB_videos_SVC.csv")
