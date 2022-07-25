
using DataFrames
using CSV
using StatsBase

ulubione = CSV.read("C:\\Users\\domi\\Desktop\\studia_sgh\\sem2\\ulubione.csv", DataFrame);
disco = CSV.read("C:\\Users\\domi\\Desktop\\studia_sgh\\sem2\\disco.csv", DataFrame);
metal = CSV.read("C:\\Users\\domi\\Desktop\\studia_sgh\\sem2\\metal.csv", DataFrame);

describe(ulubione)

ulubione[!, :danceability] = convert(Vector{Float64}, ulubione[!, :danceability]);
ulubione[!, :energy] = convert(Vector{Float64}, ulubione[!, :energy]);
ulubione[!, :speechiness] = convert(Vector{Float64}, ulubione[!, :speechiness]);
ulubione[!, :acousticness] = convert(Vector{Float64}, ulubione[!, :acousticness]);
ulubione[!, :instrumentalness] = convert(Vector{Float64}, ulubione[!, :instrumentalness]);
ulubione[!, :tempo] = convert(Vector{Float64}, ulubione[!, :tempo]);
ulubione[!, :loudness] = convert(Vector{Float64}, ulubione[!, :loudness]);
ulubione[!, :liveness] = convert(Vector{Float64}, ulubione[!, :liveness]);
ulubione[!, :valence] = convert(Vector{Float64}, ulubione[!, :valence]);

ulubione = select(ulubione, :danceability, :energy, :speechiness, :acousticness, :instrumentalness, :tempo, :loudness, :liveness, :valence);

metal[!, :danceability] = convert(Vector{Float64}, metal[!, :danceability]);
metal[!, :energy] = convert(Vector{Float64}, metal[!, :energy]);
metal[!, :speechiness] = convert(Vector{Float64}, metal[!, :speechiness]);
metal[!, :acousticness] = convert(Vector{Float64}, metal[!, :acousticness]);
metal[!, :instrumentalness] = convert(Vector{Float64}, metal[!, :instrumentalness]);
metal[!, :tempo] = convert(Vector{Float64}, metal[!, :tempo]);
metal[!, :loudness] = convert(Vector{Float64}, metal[!, :loudness]);
metal[!, :liveness] = convert(Vector{Float64}, metal[!, :liveness]);
metal[!, :valence] = convert(Vector{Float64}, metal[!, :valence]);

metal = select(metal, :danceability, :energy, :speechiness, :acousticness, :instrumentalness, :tempo, :loudness, :liveness, :valence);

disco[!, :danceability] = convert(Vector{Float64}, disco[!, :danceability]);
disco[!, :energy] = convert(Vector{Float64}, disco[!, :energy]);
disco[!, :speechiness] = convert(Vector{Float64}, disco[!, :speechiness]);
disco[!, :acousticness] = convert(Vector{Float64}, disco[!, :acousticness]);
disco[!, :instrumentalness] = convert(Vector{Float64}, disco[!, :instrumentalness]);
disco[!, :tempo] = convert(Vector{Float64}, disco[!, :tempo]);
disco[!, :loudness] = convert(Vector{Float64}, disco[!, :loudness]);
disco[!, :liveness] = convert(Vector{Float64}, disco[!, :liveness]);
disco[!, :valence] = convert(Vector{Float64}, disco[!, :valence]);

disco = select(disco, :danceability, :energy, :speechiness, :acousticness, :instrumentalness, :tempo, :loudness, :liveness, :valence);

ulubione = mapcols(zscore, ulubione);
disco = mapcols(zscore, disco);
metal = mapcols(zscore, metal);


lubiane = ulubione;

insertcols!(lubiane,
    1, 
    :like => 1.0);

first(lubiane, 5)

nrow(lubiane)

nie_lubiane = vcat(disco, metal);

insertcols!(nie_lubiane,
    1, 
    :like => 0.0);

first(nie_lubiane, 5)

nrow(nie_lubiane)

using StatsPlots

 @df lubiane cornerplot(cols(2:5))

 @df nie_lubiane cornerplot(cols(2:5))

 @df lubiane cornerplot(cols(6:10))

 @df nie_lubiane cornerplot(cols(6:10))

describe(lubiane)


describe(nie_lubiane)

df = vcat(nie_lubiane, lubiane);

using MLDataUtils
using Flux
using Statistics
using LinearAlgebra
using Metrics

using Flux: crossentropy, onecold, onehotbatch, params, train!

(X_train,y_train), (X_test,y_test) = stratifiedobs((df[:, 2:end], df[!, :like]), p = 0.7);

X_train = Matrix(X_train)';
X_test = Matrix(X_test)';

y_train = onehotbatch(y_train, 0:1)

y_test = onehotbatch(y_test, 0:1)

model = Chain(
    Dense(9, 4, relu6),
    Dense(4 ,2),
    softmax
)

loss(x, y)= Flux.binary_focal_loss(model(x), y)

loss(X_train, y_train)

ps = params(model)

opt = NADAM()

accuracy(x, y) = mean(onecold(model(x)).==onecold(y))

accuracy(X_train, y_train)

loss_history = []

epochs = 200

for epoch in 1:epochs
    train!(loss, ps, [(X_train, y_train)], opt)
    train_loss = loss(X_train, y_train)
    push!(loss_history, train_loss)
end

Plots.plot(1:epochs, loss_history,
    xlabel = "Epchos",
    ylabel = "Loss",
    title = "Learning curve",
    legend = false)

accuracy(X_train, y_train)

accuracy(X_test, y_test)

model2 = Chain(
    Dense(9, 4, relu6),
    Dense(4 ,2),
    softmax
)

loss2(x, y)= Flux.mse(model2(x), y)
loss2(X_train, y_train)

ps2 = params(model2)
opt2 = NADAM()
accuracy2(x, y) = mean(onecold(model2(x)).==onecold(y));

accuracy2(X_train, y_train)

loss_history2 = []
epochs = 400

for epoch in 1:epochs
    train!(loss2, ps2, [(X_train, y_train)], opt2)
    train_loss = loss2(X_train, y_train)
    push!(loss_history2, train_loss)
end

Plots.plot(1:epochs, loss_history2,
    xlabel = "Epchos",
    ylabel = "Loss",
    title = "Learning curve",
    legend = false)

accuracy2(X_train, y_train)

accuracy2(X_test, y_test)

model3 = Chain(
    Dense(9, 6, relu6),
    Dense(6, 4),
    Dense(4, 2),
    softmax
)

loss3(x, y)= Flux.mse(model3(x), y)
loss3(X_train, y_train)

ps3 = params(model3)
opt3 = NADAM(0.01)
accuracy3(x, y) = mean(onecold(model3(x)).==onecold(y));

accuracy3(X_train, y_train)

loss_history3 = []
epochs = 300

for epoch in 1:epochs
    train!(loss3, ps3, [(X_train, y_train)], opt3)
    train_loss = loss3(X_train, y_train)
    push!(loss_history3, train_loss)
end

Plots.plot(1:epochs, loss_history3)

accuracy3(X_train, y_train)

accuracy3(X_test, y_test)

new_music_polska_raw = CSV.read("C:\\Users\\domi\\Desktop\\studia_sgh\\sem2\\new_music_polska.csv", DataFrame);

new_music_polska = new_music_polska_raw;

new_music_polska[!, :danceability] = convert(Vector{Float64}, new_music_polska[!, :danceability]);
new_music_polska[!, :energy] = convert(Vector{Float64}, new_music_polska[!, :energy]);
new_music_polska[!, :speechiness] = convert(Vector{Float64}, new_music_polska[!, :speechiness]);
new_music_polska[!, :acousticness] = convert(Vector{Float64}, new_music_polska[!, :acousticness]);
new_music_polska[!, :instrumentalness] = convert(Vector{Float64}, new_music_polska[!, :instrumentalness]);
new_music_polska[!, :tempo] = convert(Vector{Float64}, new_music_polska[!, :tempo]);
new_music_polska[!, :loudness] = convert(Vector{Float64}, new_music_polska[!, :loudness]);
new_music_polska[!, :liveness] = convert(Vector{Float64}, new_music_polska[!, :liveness]);
new_music_polska[!, :valence] = convert(Vector{Float64}, new_music_polska[!, :valence]);

new_music_polska = select(new_music_polska, :danceability, :energy, :speechiness, :acousticness, :instrumentalness, :tempo, :loudness, :liveness, :valence);

new_music_polska = mapcols(zscore, new_music_polska);

new_music_polska = Matrix(new_music_polska)' ;

likes = model3(new_music_polska)

likes = onecold(likes).-1

likes_new_polska = hcat(new_music_polska_raw, likes);

select(filter(:x1 => ==(1), likes_new_polska), :name, :artist)

rap = CSV.read("C:\\Users\\domi\\Desktop\\studia_sgh\\sem2\\rap.csv", DataFrame);

rap[!, :danceability] = convert(Vector{Float64}, rap[!, :danceability]);
rap[!, :energy] = convert(Vector{Float64}, rap[!, :energy]);
rap[!, :speechiness] = convert(Vector{Float64}, rap[!, :speechiness]);
rap[!, :acousticness] = convert(Vector{Float64}, rap[!, :acousticness]);
rap[!, :instrumentalness] = convert(Vector{Float64}, rap[!, :instrumentalness]);
rap[!, :tempo] = convert(Vector{Float64}, rap[!, :tempo]);
rap[!, :loudness] = convert(Vector{Float64}, rap[!, :loudness]);
rap[!, :liveness] = convert(Vector{Float64}, rap[!, :liveness]);
rap[!, :valence] = convert(Vector{Float64}, rap[!, :valence]);

rap = select(rap, :danceability, :energy, :speechiness, :acousticness, :instrumentalness, :tempo, :loudness, :liveness, :valence);

rap = mapcols(zscore, rap);

rap = Matrix(rap)';

like_rap = model3(rap);

like_rap = onecold(like_rap).-1

mean(like_rap)
